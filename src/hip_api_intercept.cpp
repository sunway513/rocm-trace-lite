/*
 * hip_api_intercept.cpp — HIP Runtime API interposition via LD_PRELOAD
 *
 * Intercepts HIP API calls using dlsym(RTLD_NEXT) to forward to the real
 * implementation loaded by the host process, after recording timestamps
 * and arguments. When RTL_MODE=hip is not set, all wrappers forward
 * directly via the resolved function pointer with no recording.
 *
 * Re-entrancy guard: ScopedReentrancyGuard (RAII) prevents recursive
 * recording when HIP runtime initialization calls other HIP functions
 * internally. The guard restores the flag even on early return or
 * exception.
 *
 * Symbol resolution: std::call_once ensures each real function pointer
 * is resolved exactly once per symbol. Primary: dlsym(RTLD_NEXT, ...).
 * Fallback: dlopen("libamdhip64.so", RTLD_LAZY | RTLD_NOLOAD) guarded
 * by RTLD_NOLOAD so we never force-load HIP in processes that don't
 * already pull it in. The fallback exists because multi-process LLM
 * serving frameworks (vLLM, ATOM TP>1) fork-exec worker processes in
 * which librtl.so is LD_PRELOAD'd BEFORE libamdhip64 is brought in by
 * Python imports — early HIP wrappers otherwise see null from RTLD_NEXT
 * and the worker process shuts down fatally before any GPU work
 * dispatches. If even RTLD_NOLOAD-then-LAZY fails, the wrapper fails
 * closed and returns kHipErrorUnresolved.
 *
 * Correlation (Phase 2 — NOT YET WIRED): API→kernel linking requires
 * mapping a HIP stream pointer to the HSA queue used for dispatch. That
 * mapping must be built in hsa_intercept.cpp (hsa_queue_create callback)
 * and consumed when the completion worker processes each kernel. The
 * CorrelationMap class is kept here as a placeholder but its pop() side
 * is not invoked by hsa_intercept.cpp in this phase; tests explicitly
 * allow link_count == 0.
 *
 * Dependencies: dlsym only. NO link against libamdhip64.so, NO dlopen.
 */
#include "trace_db.h"
#include "hip_api_intercept.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <unistd.h>
#include <sys/syscall.h>

using namespace trace_db;

namespace hip_api {
    std::atomic<bool> g_enabled{false};
    CorrelationMap g_correlation_map;

    // Phase 1: push/pop are retained for ABI stability with hsa_intercept.cpp,
    // but hsa_intercept currently does not call pop(). Phase 2 will wire the
    // consumer side (see file header).
    void CorrelationMap::push(uint64_t stream, uint64_t correlation_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        map_[stream].push_back(correlation_id);
    }

    uint64_t CorrelationMap::pop(uint64_t queue_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(queue_id);
        if (it != map_.end() && !it->second.empty()) {
            uint64_t id = it->second.front();
            it->second.pop_front();
            return id;
        }
        return 0;
    }
} // namespace hip_api

// Re-entrancy guard: prevents recursive recording during HIP init.
// RAII: scope enter sets flag, scope exit restores it even on early return
// or exception — a bare `tls = true; ... tls = false;` pair is exception-
// unsafe and can deadlock a thread's tracing if any record_hip_api throws.
static thread_local bool tls_in_hip_api = false;

class ScopedReentrancyGuard {
public:
    ScopedReentrancyGuard() : prev_(tls_in_hip_api) { tls_in_hip_api = true; }
    ~ScopedReentrancyGuard() { tls_in_hip_api = prev_; }
    ScopedReentrancyGuard(const ScopedReentrancyGuard&) = delete;
    ScopedReentrancyGuard& operator=(const ScopedReentrancyGuard&) = delete;
private:
    bool prev_;
};

static int get_tid() {
    return (int)syscall(SYS_gettid);
}

static bool is_recording_ready() {
    return is_trace_ready() || trace_db::get_api_event_callback() != nullptr;
}

static void deliver_hip_api(const char* name, const char* args,
                             uint64_t start_ns, uint64_t duration_ns,
                             uint64_t correlation_id) {
    auto cb = trace_db::get_api_event_callback();
    if (cb) {
        trace_db::ApiEventRecord rec;
        rec.name = name;
        rec.args = args;
        rec.start_ns = start_ns;
        rec.end_ns = start_ns + duration_ns;
        rec.correlation_id = correlation_id;
        rec.pid = getpid();
        rec.tid = get_tid();
        cb(rec, trace_db::get_api_event_callback_data());
    } else {
        get_trace_db().record_hip_api(name, args, start_ns, duration_ns,
                                       correlation_id, getpid(), get_tid());
    }
}

// HIP type definitions (avoid including hip_runtime_api.h to keep zero-dep)
typedef int hipError_t;
typedef void* hipStream_t;
typedef void* hipDeviceptr_t;
typedef void* hipFunction_t;
typedef void* hipGraphExec_t;
typedef void* hipGraph_t;
typedef void* hipEvent_t;
typedef void* hipModule_t;

// hipErrorInvalidValue == 7 in the HIP enum. We use it as the "unresolved
// symbol" return because returning a raw literal like 1 gives an ambiguous
// error to the caller; invalid-value is the closest semantic match for
// "wrapper could not forward because the underlying HIP symbol is absent".
static const hipError_t kHipErrorUnresolved = 7;

struct hipDeviceProp_t;  // opaque, only passed by pointer

typedef enum {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

// libamdhip64 handle — lazily populated as a fork-safe fallback when
// RTLD_NEXT cannot resolve a HIP symbol in the caller's link namespace.
// This is needed for multi-process DP/TP serving frameworks (vLLM, ATOM
// DSR1 TP=4) where the Python multiprocessing spawn workers exec into a
// state in which librtl.so is LD_PRELOAD'd BEFORE libamdhip64 is loaded
// via Python imports — early wrappers then see null from RTLD_NEXT and
// the worker shuts down fatally. RTLD_NOLOAD first so we never force-
// load libamdhip64 into a process that genuinely does not use HIP; only
// if the symbol is reachable via any later-loaded HIP copy do we accept
// the dlopen fallback. Resolution still goes through std::call_once so
// publication is race-free.
static void* get_hip_handle() {
    static std::once_flag hip_handle_once;
    static void* hip_handle = nullptr;
    std::call_once(hip_handle_once, []() {
        hip_handle = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!hip_handle) {
            hip_handle = dlopen("libamdhip64.so", RTLD_LAZY);
        }
    });
    return hip_handle;
}

// Real function pointers resolved exactly once. Primary path is
// dlsym(RTLD_NEXT, ...); the fallback covers fork-exec'd children where
// RTLD_NEXT walks past our librtl.so and does not find libamdhip64 yet.
// std::call_once gives the publication barrier that a plain
// check-then-set on a raw pointer lacks (UB under the C++ memory model).
#define DECLARE_REAL(ret, name, ...) \
    typedef ret (*name##_fn)(__VA_ARGS__); \
    static name##_fn real_##name = nullptr; \
    static std::once_flag resolve_##name##_flag; \
    static void resolve_##name() { \
        std::call_once(resolve_##name##_flag, []() { \
            real_##name = (name##_fn)dlsym(RTLD_NEXT, #name); \
            if (!real_##name) { \
                void* h = get_hip_handle(); \
                if (h) real_##name = (name##_fn)dlsym(h, #name); \
            } \
        }); \
    }

DECLARE_REAL(hipError_t, hipModuleLaunchKernel,
    hipFunction_t, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, hipStream_t, void**, void**)

DECLARE_REAL(hipError_t, hipExtModuleLaunchKernel,
    hipFunction_t, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, hipStream_t, void**, void**,
    void*, void*, unsigned int)

DECLARE_REAL(hipError_t, hipMemcpy,
    void*, const void*, size_t, hipMemcpyKind)

DECLARE_REAL(hipError_t, hipMemcpyAsync,
    void*, const void*, size_t, hipMemcpyKind, hipStream_t)

DECLARE_REAL(hipError_t, hipMalloc, void**, size_t)
DECLARE_REAL(hipError_t, hipFree, void*)

DECLARE_REAL(hipError_t, hipStreamSynchronize, hipStream_t)
DECLARE_REAL(hipError_t, hipDeviceSynchronize)

DECLARE_REAL(hipError_t, hipGraphLaunch, hipGraphExec_t, hipStream_t)

// Low-frequency APIs (zero overhead tier — called a few times per process)
DECLARE_REAL(hipError_t, hipSetDevice, int)
DECLARE_REAL(hipError_t, hipStreamCreate, hipStream_t*)
DECLARE_REAL(hipError_t, hipStreamDestroy, hipStream_t)
DECLARE_REAL(hipError_t, hipEventCreate, hipEvent_t*)
DECLARE_REAL(hipError_t, hipEventDestroy, hipEvent_t)
DECLARE_REAL(hipError_t, hipEventRecord, hipEvent_t, hipStream_t)
DECLARE_REAL(hipError_t, hipEventSynchronize, hipEvent_t)
DECLARE_REAL(hipError_t, hipGraphCreate, hipGraph_t*, unsigned int)
DECLARE_REAL(hipError_t, hipGraphInstantiate, hipGraphExec_t*, hipGraph_t, void*, void*, size_t)
DECLARE_REAL(hipError_t, hipGraphExecDestroy, hipGraphExec_t)
DECLARE_REAL(hipError_t, hipHostMalloc, void**, size_t, unsigned int)
DECLARE_REAL(hipError_t, hipHostFree, void*)

// ---- HIP API Wrappers ----

extern "C" {

hipError_t hipModuleLaunchKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes,
    hipStream_t stream, void** kernelParams, void** extra) {

    resolve_hipModuleLaunchKernel();
    if (!real_hipModuleLaunchKernel) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream,
            kernelParams, extra);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();

    // Phase 2 TODO: once hsa_intercept.cpp wires the stream->queue map,
    // push (stream, corr) here so the HSA completion worker can link the op.

    hipError_t ret = real_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream,
        kernelParams, extra);

    uint64_t t1 = tick();
    char args[128];
    snprintf(args, sizeof(args), "grid=%u,%u,%u block=%u,%u,%u shared=%u",
             gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
             sharedMemBytes);
    deliver_hip_api("hipModuleLaunchKernel", args,
                                  t0, t1 - t0, corr);
    return ret;
}

hipError_t hipExtModuleLaunchKernel(
    hipFunction_t f, unsigned int globalWorkSizeX, unsigned int globalWorkSizeY,
    unsigned int globalWorkSizeZ, unsigned int localWorkSizeX,
    unsigned int localWorkSizeY, unsigned int localWorkSizeZ,
    unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra,
    void* startEvent, void* stopEvent, unsigned int flags) {

    resolve_hipExtModuleLaunchKernel();
    if (!real_hipExtModuleLaunchKernel) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipExtModuleLaunchKernel(f, globalWorkSizeX, globalWorkSizeY,
            globalWorkSizeZ, localWorkSizeX, localWorkSizeY, localWorkSizeZ,
            sharedMemBytes, stream, kernelParams, extra,
            startEvent, stopEvent, flags);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();

    // Phase 2 TODO: once hsa_intercept.cpp wires the stream->queue map,
    // push (stream, corr) here so the HSA completion worker can link the op.

    hipError_t ret = real_hipExtModuleLaunchKernel(f, globalWorkSizeX,
        globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
        localWorkSizeZ, sharedMemBytes, stream, kernelParams, extra,
        startEvent, stopEvent, flags);

    uint64_t t1 = tick();
    char args[128];
    snprintf(args, sizeof(args), "grid=%u,%u,%u block=%u,%u,%u shared=%u",
             globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
             localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes);
    deliver_hip_api("hipExtModuleLaunchKernel", args,
                                  t0, t1 - t0, corr);
    return ret;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                     hipMemcpyKind kind) {
    resolve_hipMemcpy();
    if (!real_hipMemcpy) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipMemcpy(dst, src, sizeBytes, kind);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipMemcpy(dst, src, sizeBytes, kind);
    uint64_t t1 = tick();
    char args[64];
    snprintf(args, sizeof(args), "size=%zu kind=%d", sizeBytes, (int)kind);
    deliver_hip_api("hipMemcpy", args, t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {
    resolve_hipMemcpyAsync();
    if (!real_hipMemcpyAsync) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
    uint64_t t1 = tick();
    char args[64];
    snprintf(args, sizeof(args), "size=%zu kind=%d", sizeBytes, (int)kind);
    deliver_hip_api("hipMemcpyAsync", args, t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipMalloc(void** ptr, size_t size) {
    resolve_hipMalloc();
    if (!real_hipMalloc) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipMalloc(ptr, size);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipMalloc(ptr, size);
    uint64_t t1 = tick();
    char args[64];
    snprintf(args, sizeof(args), "size=%zu", size);
    deliver_hip_api("hipMalloc", args, t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipFree(void* ptr) {
    resolve_hipFree();
    if (!real_hipFree) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipFree(ptr);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipFree(ptr);
    uint64_t t1 = tick();
    deliver_hip_api("hipFree", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
    resolve_hipStreamSynchronize();
    if (!real_hipStreamSynchronize) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipStreamSynchronize(stream);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipStreamSynchronize(stream);
    uint64_t t1 = tick();
    deliver_hip_api("hipStreamSynchronize", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipDeviceSynchronize() {
    resolve_hipDeviceSynchronize();
    if (!real_hipDeviceSynchronize) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipDeviceSynchronize();
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipDeviceSynchronize();
    uint64_t t1 = tick();
    deliver_hip_api("hipDeviceSynchronize", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
    resolve_hipGraphLaunch();
    if (!real_hipGraphLaunch) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipGraphLaunch(graphExec, stream);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();

    // Phase 2 TODO: once hsa_intercept.cpp wires the stream->queue map,
    // push (stream, corr) here so the HSA completion worker can link the op.

    hipError_t ret = real_hipGraphLaunch(graphExec, stream);
    uint64_t t1 = tick();
    deliver_hip_api("hipGraphLaunch", "", t0, t1 - t0,
                                  corr);
    return ret;
}

// ---- Low-frequency API Wrappers (zero overhead tier) ----

hipError_t hipSetDevice(int deviceId) {
    resolve_hipSetDevice();
    if (!real_hipSetDevice) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipSetDevice(deviceId);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipSetDevice(deviceId);
    uint64_t t1 = tick();
    char args[32];
    snprintf(args, sizeof(args), "device=%d", deviceId);
    deliver_hip_api("hipSetDevice", args, t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipStreamCreate(hipStream_t* stream) {
    resolve_hipStreamCreate();
    if (!real_hipStreamCreate) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipStreamCreate(stream);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipStreamCreate(stream);
    uint64_t t1 = tick();
    deliver_hip_api("hipStreamCreate", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipStreamDestroy(hipStream_t stream) {
    resolve_hipStreamDestroy();
    if (!real_hipStreamDestroy) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipStreamDestroy(stream);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipStreamDestroy(stream);
    uint64_t t1 = tick();
    deliver_hip_api("hipStreamDestroy", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipEventCreate(hipEvent_t* event) {
    resolve_hipEventCreate();
    if (!real_hipEventCreate) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipEventCreate(event);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipEventCreate(event);
    uint64_t t1 = tick();
    deliver_hip_api("hipEventCreate", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipEventDestroy(hipEvent_t event) {
    resolve_hipEventDestroy();
    if (!real_hipEventDestroy) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipEventDestroy(event);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipEventDestroy(event);
    uint64_t t1 = tick();
    deliver_hip_api("hipEventDestroy", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    resolve_hipEventRecord();
    if (!real_hipEventRecord) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipEventRecord(event, stream);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipEventRecord(event, stream);
    uint64_t t1 = tick();
    deliver_hip_api("hipEventRecord", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipEventSynchronize(hipEvent_t event) {
    resolve_hipEventSynchronize();
    if (!real_hipEventSynchronize) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipEventSynchronize(event);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipEventSynchronize(event);
    uint64_t t1 = tick();
    deliver_hip_api("hipEventSynchronize", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipGraphCreate(hipGraph_t* graph, unsigned int flags) {
    resolve_hipGraphCreate();
    if (!real_hipGraphCreate) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipGraphCreate(graph, flags);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipGraphCreate(graph, flags);
    uint64_t t1 = tick();
    deliver_hip_api("hipGraphCreate", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                               void* errNode, void* errLog, size_t bufSize) {
    resolve_hipGraphInstantiate();
    if (!real_hipGraphInstantiate) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipGraphInstantiate(exec, graph, errNode, errLog, bufSize);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipGraphInstantiate(exec, graph, errNode, errLog, bufSize);
    uint64_t t1 = tick();
    deliver_hip_api("hipGraphInstantiate", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipGraphExecDestroy(hipGraphExec_t exec) {
    resolve_hipGraphExecDestroy();
    if (!real_hipGraphExecDestroy) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipGraphExecDestroy(exec);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipGraphExecDestroy(exec);
    uint64_t t1 = tick();
    deliver_hip_api("hipGraphExecDestroy", "", t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
    resolve_hipHostMalloc();
    if (!real_hipHostMalloc) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipHostMalloc(ptr, size, flags);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipHostMalloc(ptr, size, flags);
    uint64_t t1 = tick();
    char args[64];
    snprintf(args, sizeof(args), "size=%zu flags=%u", size, flags);
    deliver_hip_api("hipHostMalloc", args, t0, t1 - t0,
                                  corr);
    return ret;
}

hipError_t hipHostFree(void* ptr) {
    resolve_hipHostFree();
    if (!real_hipHostFree) return kHipErrorUnresolved;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_recording_ready()) {
        return real_hipHostFree(ptr);
    }
    ScopedReentrancyGuard _guard;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipHostFree(ptr);
    uint64_t t1 = tick();
    deliver_hip_api("hipHostFree", "", t0, t1 - t0,
                                  corr);
    return ret;
}

} // extern "C"
