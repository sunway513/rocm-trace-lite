/*
 * hip_api_intercept.cpp — HIP Runtime API interposition via LD_PRELOAD
 *
 * Intercepts HIP API calls using dlsym(RTLD_NEXT) to forward to real
 * implementation after recording timestamps and arguments.
 *
 * Re-entrancy guard: thread-local boolean prevents recursive recording
 * when HIP runtime initialization calls other HIP functions internally.
 *
 * Correlation: each API call gets a unique correlation_id pushed to a
 * per-stream queue. The HSA completion worker in hsa_intercept.cpp pops
 * from this queue to link GPU kernel dispatches back to their API calls.
 *
 * Activated by RTL_MODE=hip. When disabled, all wrappers forward directly.
 *
 * Dependencies: dlsym only. NO link against libamdhip64.so.
 */
#include "trace_db.h"
#include "hip_api_intercept.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <sys/syscall.h>

using namespace trace_db;

namespace hip_api {
    std::atomic<bool> g_enabled{false};
    CorrelationMap g_correlation_map;

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

// Re-entrancy guard: prevents recursive recording during HIP init
static thread_local bool tls_in_hip_api = false;

static int get_tid() {
    return (int)syscall(SYS_gettid);
}

// Macro for wrapper boilerplate
#define HIP_WRAPPER_PROLOG(fname) \
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed) \
        || !is_trace_ready()) { \
        return real_##fname; \
    } \
    tls_in_hip_api = true; \
    uint64_t _corr = next_correlation_id(); \
    uint64_t _t0 = tick();

#define HIP_WRAPPER_EPILOG(fname, args_str) \
    uint64_t _t1 = tick(); \
    get_trace_db().record_hip_api(#fname, args_str, _t0, _t1 - _t0, \
                                  _corr, getpid(), get_tid()); \
    tls_in_hip_api = false;

// HIP type definitions (avoid including hip_runtime_api.h to keep zero-dep)
typedef int hipError_t;
typedef void* hipStream_t;
typedef void* hipDeviceptr_t;
typedef void* hipFunction_t;
typedef void* hipGraphExec_t;
typedef void* hipGraph_t;
typedef void* hipEvent_t;
typedef void* hipModule_t;

struct hipDeviceProp_t;  // opaque, only passed by pointer

typedef enum {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

// Real function pointers (resolved lazily via dlsym)
static void* g_hip_lib_handle = nullptr;
static void ensure_hip_handle() {
    if (!g_hip_lib_handle) {
        g_hip_lib_handle = dlopen("libamdhip64.so", RTLD_NOW | RTLD_NOLOAD);
        if (!g_hip_lib_handle) {
            g_hip_lib_handle = dlopen("libamdhip64.so", RTLD_NOW);
        }
    }
}

#define DECLARE_REAL(ret, name, ...) \
    typedef ret (*name##_fn)(__VA_ARGS__); \
    static name##_fn real_##name = nullptr; \
    static void resolve_##name() { \
        if (!real_##name) { \
            ensure_hip_handle(); \
            if (g_hip_lib_handle) \
                real_##name = (name##_fn)dlsym(g_hip_lib_handle, #name); \
            if (!real_##name) \
                real_##name = (name##_fn)dlsym(RTLD_NEXT, #name); \
        } \
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
    if (!real_hipModuleLaunchKernel) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream,
            kernelParams, extra);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();

    hip_api::g_correlation_map.push((uint64_t)stream, corr);

    hipError_t ret = real_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream,
        kernelParams, extra);

    uint64_t t1 = tick();
    char args[128];
    snprintf(args, sizeof(args), "grid=%u,%u,%u block=%u,%u,%u shared=%u",
             gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
             sharedMemBytes);
    get_trace_db().record_hip_api("hipModuleLaunchKernel", args,
                                  t0, t1 - t0, corr, getpid(), get_tid());
    tls_in_hip_api = false;
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
    if (!real_hipExtModuleLaunchKernel) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipExtModuleLaunchKernel(f, globalWorkSizeX, globalWorkSizeY,
            globalWorkSizeZ, localWorkSizeX, localWorkSizeY, localWorkSizeZ,
            sharedMemBytes, stream, kernelParams, extra,
            startEvent, stopEvent, flags);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();

    hip_api::g_correlation_map.push((uint64_t)stream, corr);

    hipError_t ret = real_hipExtModuleLaunchKernel(f, globalWorkSizeX,
        globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
        localWorkSizeZ, sharedMemBytes, stream, kernelParams, extra,
        startEvent, stopEvent, flags);

    uint64_t t1 = tick();
    char args[128];
    snprintf(args, sizeof(args), "grid=%u,%u,%u block=%u,%u,%u shared=%u",
             globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
             localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes);
    get_trace_db().record_hip_api("hipExtModuleLaunchKernel", args,
                                  t0, t1 - t0, corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                     hipMemcpyKind kind) {
    resolve_hipMemcpy();
    if (!real_hipMemcpy) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipMemcpy(dst, src, sizeBytes, kind);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipMemcpy(dst, src, sizeBytes, kind);
    uint64_t t1 = tick();
    char args[64];
    snprintf(args, sizeof(args), "size=%zu kind=%d", sizeBytes, (int)kind);
    get_trace_db().record_hip_api("hipMemcpy", args, t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {
    resolve_hipMemcpyAsync();
    if (!real_hipMemcpyAsync) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
    uint64_t t1 = tick();
    char args[64];
    snprintf(args, sizeof(args), "size=%zu kind=%d", sizeBytes, (int)kind);
    get_trace_db().record_hip_api("hipMemcpyAsync", args, t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipMalloc(void** ptr, size_t size) {
    resolve_hipMalloc();
    if (!real_hipMalloc) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipMalloc(ptr, size);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipMalloc(ptr, size);
    uint64_t t1 = tick();
    char args[64];
    snprintf(args, sizeof(args), "size=%zu", size);
    get_trace_db().record_hip_api("hipMalloc", args, t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipFree(void* ptr) {
    resolve_hipFree();
    if (!real_hipFree) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipFree(ptr);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipFree(ptr);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipFree", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
    resolve_hipStreamSynchronize();
    if (!real_hipStreamSynchronize) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipStreamSynchronize(stream);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipStreamSynchronize(stream);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipStreamSynchronize", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipDeviceSynchronize() {
    resolve_hipDeviceSynchronize();
    if (!real_hipDeviceSynchronize) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipDeviceSynchronize();
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipDeviceSynchronize();
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipDeviceSynchronize", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
    resolve_hipGraphLaunch();
    if (!real_hipGraphLaunch) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipGraphLaunch(graphExec, stream);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();

    hip_api::g_correlation_map.push((uint64_t)stream, corr);

    hipError_t ret = real_hipGraphLaunch(graphExec, stream);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipGraphLaunch", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

// ---- Low-frequency API Wrappers (zero overhead tier) ----

hipError_t hipSetDevice(int deviceId) {
    resolve_hipSetDevice();
    if (!real_hipSetDevice) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipSetDevice(deviceId);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipSetDevice(deviceId);
    uint64_t t1 = tick();
    char args[32];
    snprintf(args, sizeof(args), "device=%d", deviceId);
    get_trace_db().record_hip_api("hipSetDevice", args, t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipStreamCreate(hipStream_t* stream) {
    resolve_hipStreamCreate();
    if (!real_hipStreamCreate) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipStreamCreate(stream);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipStreamCreate(stream);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipStreamCreate", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipStreamDestroy(hipStream_t stream) {
    resolve_hipStreamDestroy();
    if (!real_hipStreamDestroy) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipStreamDestroy(stream);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipStreamDestroy(stream);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipStreamDestroy", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipEventCreate(hipEvent_t* event) {
    resolve_hipEventCreate();
    if (!real_hipEventCreate) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipEventCreate(event);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipEventCreate(event);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipEventCreate", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipEventDestroy(hipEvent_t event) {
    resolve_hipEventDestroy();
    if (!real_hipEventDestroy) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipEventDestroy(event);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipEventDestroy(event);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipEventDestroy", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    resolve_hipEventRecord();
    if (!real_hipEventRecord) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipEventRecord(event, stream);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipEventRecord(event, stream);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipEventRecord", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipEventSynchronize(hipEvent_t event) {
    resolve_hipEventSynchronize();
    if (!real_hipEventSynchronize) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipEventSynchronize(event);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipEventSynchronize(event);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipEventSynchronize", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipGraphCreate(hipGraph_t* graph, unsigned int flags) {
    resolve_hipGraphCreate();
    if (!real_hipGraphCreate) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipGraphCreate(graph, flags);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipGraphCreate(graph, flags);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipGraphCreate", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                               void* errNode, void* errLog, size_t bufSize) {
    resolve_hipGraphInstantiate();
    if (!real_hipGraphInstantiate) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipGraphInstantiate(exec, graph, errNode, errLog, bufSize);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipGraphInstantiate(exec, graph, errNode, errLog, bufSize);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipGraphInstantiate", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipGraphExecDestroy(hipGraphExec_t exec) {
    resolve_hipGraphExecDestroy();
    if (!real_hipGraphExecDestroy) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipGraphExecDestroy(exec);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipGraphExecDestroy(exec);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipGraphExecDestroy", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
    resolve_hipHostMalloc();
    if (!real_hipHostMalloc) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipHostMalloc(ptr, size, flags);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipHostMalloc(ptr, size, flags);
    uint64_t t1 = tick();
    char args[64];
    snprintf(args, sizeof(args), "size=%zu flags=%u", size, flags);
    get_trace_db().record_hip_api("hipHostMalloc", args, t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

hipError_t hipHostFree(void* ptr) {
    resolve_hipHostFree();
    if (!real_hipHostFree) return 1;
    if (tls_in_hip_api || !hip_api::g_enabled.load(std::memory_order_relaxed)
        || !is_trace_ready()) {
        return real_hipHostFree(ptr);
    }
    tls_in_hip_api = true;
    uint64_t corr = next_correlation_id();
    uint64_t t0 = tick();
    hipError_t ret = real_hipHostFree(ptr);
    uint64_t t1 = tick();
    get_trace_db().record_hip_api("hipHostFree", "", t0, t1 - t0,
                                  corr, getpid(), get_tid());
    tls_in_hip_api = false;
    return ret;
}

} // extern "C"
