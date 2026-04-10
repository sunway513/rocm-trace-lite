/*
 * hip_intercept.cpp -- HIP CLR profiler integration (RTL_MODE=hip)
 *
 * This path uses the HIP profiler extension API (hipProfiler*Ext) instead
 * of HSA signal injection. It requires an upstream ROCm build that includes
 * ROCm/rocm-systems branch amd/dev/gandryey/ROCM-1667-12 or later. On older
 * ROCm, the dlsym probe fails gracefully and reports 0 records.
 *
 * Activation flow:
 *   1. Python launcher sets GPU_CLR_PROFILE=1 when --mode=hip
 *   2. libamdhip64.so initializes, hip::init() calls HipProfilerInitExt()
 *      which sees GPU_CLR_PROFILE env var and installs 510 dispatch table
 *      wrappers + activity callback on ACTIVITY_DOMAIN_HIP_OPS
 *   3. App runs: each HIP API call goes through wrapper that records CPU
 *      start/end + sets correlation_id TLS; GPU activity callback writes
 *      GPU timestamps into the same record slot via atomic correlation_id
 *   4. At shutdown, hsa_intercept.cpp calls hip_profiler_drain()
 *   5. hip_profiler_drain() calls Disable -> GetRecords -> iterate -> Reset
 *   6. CLR's HipProfilerFinalizer static destructor may still run, but
 *      Reset() emptied the buffer so it writes nothing useful
 *
 * Why GPU_CLR_PROFILE=1 instead of hipProfilerEnableExt():
 *   - librtl.so is loaded by HSA runtime during hip::init(), BEFORE HIP is
 *     fully loaded. Calling hipProfilerEnableExt() from our OnLoad would
 *     fail (symbol not yet resolvable).
 *   - Setting GPU_CLR_PROFILE=1 in the environment lets the CLR profiler
 *     self-activate at the correct point in hip::init() (HipProfilerInitExt
 *     checks this env var and installs dispatch table wrappers if set).
 *
 * Dependency: libdl (already linked via Makefile). No compile-time dependency
 * on libamdhip64. All HIP profiler symbols are resolved at runtime via dlsym.
 */
#include "trace_db.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <dlfcn.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

namespace hip_intercept {

// ============================================================
// Local copies of hip_profiler_ext.h types.
//
// We cannot include the upstream header because it is not yet shipped in
// mainline ROCm. Once the header is available we can switch to including
// it directly. Layout must stay in sync with ROCm/rocm-systems branch
// amd/dev/gandryey/ROCM-1667-12 (commit 6025cd1 or later).
//
// Key design: fixed-size records (HipGpuActivityExt=128B, HipApiRecordExt=256B)
// with reserved padding for future extension without ABI breaks.
// ============================================================

enum HipGpuOpExt : uint32_t {
    OP_DISPATCH = 0,
    OP_COPY     = 1,
    OP_BARRIER  = 2,
};

struct HipGpuActivityExt {
    union {
        uint64_t _flags_u64;
        struct {
            uint64_t op        : 3;   // HipGpuOpExt
            uint64_t is_graph  : 1;   // set when op was launched from a HIP graph
            uint64_t           : 12;  // reserved
            uint64_t device_id : 16;  // device index
            uint64_t queue_id  : 16;  // queue/stream index
            uint64_t           : 16;  // reserved
        };
    };
    uint64_t    begin_ns;         // GPU begin timestamp (ns)
    uint64_t    end_ns;           // GPU end timestamp (ns)
    union {
        uint64_t    bytes;        // copy ops (OP_COPY)
        const char* kernel_name;  // dispatch ops (OP_DISPATCH, may be NULL)
    };
    uint32_t    gpu_op_count;     // total GPU ops (0=none, 1=single, >1=graph)
    uint32_t    _reserved_u32;
    const HipGpuActivityExt* gpu_ops;  // gpu_ops[0..gpu_op_count-1]
    uint8_t     _pad1[80];        // reserved
};
static_assert(sizeof(HipGpuActivityExt) == 128, "HipGpuActivityExt must be 128 bytes");

struct HipApiRecordExt {
    // CPU call info (first 128-byte half)
    const char*  api_name;        // points into DLL's API name table, never NULL
    union {
        uint64_t _flags_u64;
        struct {
            uint64_t has_gpu_activity : 1;
            uint64_t                  : 63;
        };
    };
    uint64_t     thread_id;       // hash of std::thread::id
    uint64_t     start_ns;        // CPU call begin (ns)
    uint64_t     end_ns;          // CPU call end (ns)
    void*        stream;          // hipStream_t (opaque pointer)
    uint8_t      _pad1[80];       // reserved
    // GPU activity (second 128-byte half, valid when has_gpu_activity != 0)
    HipGpuActivityExt gpu;
};
static_assert(sizeof(HipApiRecordExt) == 256, "HipApiRecordExt must be 256 bytes");

// hipError_t is 32-bit; we treat it as int to avoid pulling hip headers.
using hipProfilerEnableExt_fn     = int (*)(void);
using hipProfilerDisableExt_fn    = int (*)(void);
using hipProfilerGetRecordsExt_fn = int (*)(const HipApiRecordExt**, size_t*);
using hipProfilerResetExt_fn      = int (*)(void);

// ============================================================
// Runtime dlopen/dlsym state
// ============================================================
static void* g_hip_handle = nullptr;
static hipProfilerEnableExt_fn     g_fn_enable      = nullptr;
static hipProfilerDisableExt_fn    g_fn_disable     = nullptr;
static hipProfilerGetRecordsExt_fn g_fn_get_records = nullptr;
static hipProfilerResetExt_fn      g_fn_reset       = nullptr;
static bool g_api_ready = false;

// ============================================================
// Helper: write a single GPU activity record to the trace DB.
// ============================================================
static void write_gpu_op(trace_db::TraceDB& db,
                         const HipGpuActivityExt& gpu,
                         uint64_t corr_id,
                         uint64_t& n_kernel,
                         uint64_t& n_copy,
                         uint64_t& n_skipped) {
    if (gpu.end_ns <= gpu.begin_ns) {
        n_skipped++;
        return;
    }

    switch (gpu.op) {
        case OP_DISPATCH: {
            const char* name = gpu.kernel_name ? gpu.kernel_name : "unknown_kernel";
            db.record_kernel(name, (int)gpu.device_id, gpu.queue_id,
                             gpu.begin_ns, gpu.end_ns, corr_id);
            n_kernel++;
            break;
        }
        case OP_COPY: {
            db.record_copy((int)gpu.device_id, -1, (size_t)gpu.bytes,
                           gpu.begin_ns, gpu.end_ns, corr_id);
            n_copy++;
            break;
        }
        case OP_BARRIER: {
            db.record_kernel("Barrier", (int)gpu.device_id, gpu.queue_id,
                             gpu.begin_ns, gpu.end_ns, corr_id);
            n_kernel++;
            break;
        }
        default:
            n_skipped++;
            break;
    }
}

// ============================================================
// Probe libamdhip64.so and resolve the 4 profiler symbols.
// Returns true if all symbols are available (feature is ready).
// Uses RTLD_NOLOAD so we do not force-load HIP if the app never used it.
//
// Symbol resolution order: try new *Ext names first (ROCM-1667-12),
// then fall back to old hipClrProfiler* names (5dc10a8).
// ============================================================
bool hip_profiler_probe() {
    if (g_api_ready) return true;

    // RTLD_NOLOAD: only succeed if libamdhip64.so is already resident.
    g_hip_handle = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!g_hip_handle) {
        g_hip_handle = dlopen("libamdhip64.so.7", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (!g_hip_handle) {
        g_hip_handle = dlopen("libamdhip64.so.6", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (!g_hip_handle) {
        fprintf(stderr, "rtl[hip]: libamdhip64.so not loaded, nothing to drain\n");
        return false;
    }

    dlerror();

    // Try new *Ext symbols first (ROCM-1667-12+)
    g_fn_enable = reinterpret_cast<hipProfilerEnableExt_fn>(
        dlsym(g_hip_handle, "hipProfilerEnableExt"));
    g_fn_disable = reinterpret_cast<hipProfilerDisableExt_fn>(
        dlsym(g_hip_handle, "hipProfilerDisableExt"));
    g_fn_get_records = reinterpret_cast<hipProfilerGetRecordsExt_fn>(
        dlsym(g_hip_handle, "hipProfilerGetRecordsExt"));
    g_fn_reset = reinterpret_cast<hipProfilerResetExt_fn>(
        dlsym(g_hip_handle, "hipProfilerResetExt"));

    if (g_fn_enable && g_fn_disable && g_fn_get_records && g_fn_reset) {
        g_api_ready = true;
        fprintf(stderr, "rtl[hip]: hipProfiler*Ext API resolved, drain on exit enabled\n");
        return true;
    }

    // Fall back to old hipClrProfiler* symbols (initial 5dc10a8)
    g_fn_enable = reinterpret_cast<hipProfilerEnableExt_fn>(
        dlsym(g_hip_handle, "hipClrProfilerEnable"));
    g_fn_disable = reinterpret_cast<hipProfilerDisableExt_fn>(
        dlsym(g_hip_handle, "hipClrProfilerDisable"));
    g_fn_get_records = reinterpret_cast<hipProfilerGetRecordsExt_fn>(
        dlsym(g_hip_handle, "hipClrProfilerGetRecords"));
    g_fn_reset = reinterpret_cast<hipProfilerResetExt_fn>(
        dlsym(g_hip_handle, "hipClrProfilerReset"));

    if (g_fn_enable && g_fn_disable && g_fn_get_records && g_fn_reset) {
        g_api_ready = true;
        fprintf(stderr, "rtl[hip]: hipClrProfiler API resolved (legacy names), drain on exit enabled\n");
        return true;
    }

    fprintf(stderr,
            "rtl[hip]: hipProfiler API not available in this ROCm build\n"
            "rtl[hip]:   enable=%p disable=%p get_records=%p reset=%p\n"
            "rtl[hip]: requires ROCm build with rocm-systems ROCM-1667-12 branch or later\n"
            "rtl[hip]: this process will record nothing\n",
            (void*)g_fn_enable, (void*)g_fn_disable,
            (void*)g_fn_get_records, (void*)g_fn_reset);
    dlclose(g_hip_handle);
    g_hip_handle = nullptr;
    return false;
}

// ============================================================
// Drain all accumulated HIP profiler records into the trace database.
// Called from hsa_intercept::shutdown() (atexit handler).
//
// Ordering:
//   - Disable() flushes in-flight GPU work. All records are finalized.
//   - GetRecords() returns the full array. Safe at shutdown (no new HIP calls).
//   - Reset() frees the CLR-side buffer.
//
// The CLR profiler's HipProfilerFinalizer may still run later (static
// destructor order), but Reset() empties the record buffer before it runs,
// so the finalizer has nothing to write.
// ============================================================
void hip_profiler_drain() {
    if (!hip_profiler_probe()) return;

    int rc = g_fn_disable();
    if (rc != 0) {
        fprintf(stderr, "rtl[hip]: hipProfilerDisableExt returned %d\n", rc);
    }

    const HipApiRecordExt* records = nullptr;
    size_t count = 0;
    rc = g_fn_get_records(&records, &count);
    if (rc != 0 || !records) {
        fprintf(stderr, "rtl[hip]: hipProfilerGetRecordsExt returned %d (count=%zu)\n",
                rc, count);
        return;
    }

    fprintf(stderr, "rtl[hip]: draining %zu records from HIP profiler\n", count);

    const int pid = (int)getpid();
    auto& db = trace_db::get_trace_db();
    uint64_t n_kernel = 0, n_copy = 0, n_api = 0, n_graph_ops = 0, n_skipped = 0;

    for (size_t i = 0; i < count; i++) {
        const HipApiRecordExt& rec = records[i];
        const uint64_t corr_id = (uint64_t)i;

        // API name is a direct pointer in the new API (never NULL).
        const char* api_name = rec.api_name ? rec.api_name : "HipApi";

        // Write CPU-side HIP API record.
        if (rec.end_ns > rec.start_ns) {
            db.record_hip_api(api_name, /*args*/ "",
                              rec.start_ns,
                              rec.end_ns - rec.start_ns,
                              corr_id,
                              pid,
                              (int)rec.thread_id);
            n_api++;
        } else {
            n_skipped++;
        }

        // Write GPU-side activity records.
        if (!rec.has_gpu_activity) continue;

        // Graph launches may have multiple GPU ops (gpu_op_count > 1).
        // For single ops, gpu_ops points to the embedded gpu field itself.
        if (rec.gpu.gpu_op_count > 1 && rec.gpu.gpu_ops) {
            for (uint32_t j = 0; j < rec.gpu.gpu_op_count; j++) {
                write_gpu_op(db, rec.gpu.gpu_ops[j], corr_id,
                             n_kernel, n_copy, n_skipped);
                n_graph_ops++;
            }
        } else {
            write_gpu_op(db, rec.gpu, corr_id,
                         n_kernel, n_copy, n_skipped);
        }
    }

    fprintf(stderr, "rtl[hip]: drained api=%" PRIu64 " kernel=%" PRIu64
            " copy=%" PRIu64 " graph_ops=%" PRIu64 " skipped=%" PRIu64 "\n",
            n_api, n_kernel, n_copy, n_graph_ops, n_skipped);

    g_fn_reset();
}

} // namespace hip_intercept
