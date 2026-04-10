/*
 * hip_intercept.cpp — HIP CLR profiler integration (RTL_MODE=hip)
 *
 * This path uses the HIP CLR built-in profiler API (hipClrProfiler*) instead
 * of HSA signal injection. It requires an upstream ROCm build that includes
 * ROCm/rocm-systems@5dc10a8 or later. On older ROCm, the dlsym probe fails
 * and the caller falls back to RTL_MODE=default.
 *
 * Activation flow:
 *   1. Python launcher sets GPU_CLR_PROFILE=/dev/null when --mode=hip
 *   2. libamdhip64.so initializes → HipClrProfilerInit() sees the env var →
 *      activates dispatch table wrappers + activity callback
 *   3. App runs → CLR profiler accumulates records in its internal buffer
 *   4. At shutdown, hsa_intercept.cpp calls hip_profiler_drain()
 *   5. hip_profiler_drain() calls Disable → GetRecords → iterate → Reset
 *
 * Why GPU_CLR_PROFILE=/dev/null instead of hipClrProfilerEnable():
 *   - librtl.so is loaded by HSA runtime during hip::init(), BEFORE HIP is
 *     fully loaded. Calling hipClrProfilerEnable() from our OnLoad would
 *     fail (symbol not yet resolvable).
 *   - Setting GPU_CLR_PROFILE in the environment lets the CLR profiler
 *     self-activate at the correct point in hip::init().
 *   - /dev/null suppresses the CLR profiler's own JSON autosave; we extract
 *     records via hipClrProfilerGetRecords() and write our own SQLite format.
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
// Local copies of hip_clr_profiler_ext.h types.
//
// We cannot include the upstream header because it is not yet shipped in
// mainline ROCm. Once the header is available we can switch to including
// it directly. Layout must stay in sync with ROCm/rocm-systems@5dc10a8.
// ============================================================

struct HipClrGpuActivity {
    uint32_t    op;           // 0=dispatch, 1=copy, 2=barrier
    uint64_t    begin_ns;     // GPU begin timestamp (ns)
    uint64_t    end_ns;       // GPU end timestamp (ns)
    int         device_id;
    uint64_t    queue_id;
    uint64_t    bytes;        // copy ops
    const char* kernel_name;  // dispatch ops (may be NULL)
};

struct HipClrApiRecord {
    uint32_t          api_id;
    uint64_t          thread_id;
    uint64_t          start_ns;         // CPU timestamp
    uint64_t          end_ns;           // CPU timestamp
    int               has_gpu_activity;
    HipClrGpuActivity gpu;
};

enum HipClrOp : uint32_t {
    OP_DISPATCH = 0,
    OP_COPY     = 1,
    OP_BARRIER  = 2,
};

// hipError_t is 32-bit; we treat it as int to avoid pulling hip headers.
using hipClrProfilerEnable_fn    = int (*)(void);
using hipClrProfilerDisable_fn   = int (*)(void);
using hipClrProfilerGetRecords_fn = int (*)(const HipClrApiRecord**, size_t*);
using hipClrProfilerReset_fn     = int (*)(void);

// ============================================================
// Runtime dlopen/dlsym state
// ============================================================
static void* g_hip_handle = nullptr;
static hipClrProfilerEnable_fn     g_fn_enable     = nullptr;
static hipClrProfilerDisable_fn    g_fn_disable    = nullptr;
static hipClrProfilerGetRecords_fn g_fn_get_records = nullptr;
static hipClrProfilerReset_fn      g_fn_reset      = nullptr;
static bool g_api_ready = false;

// Dispatch table wrappers do not expose an API name table yet.
// Until the upstream header ships kHipClrApiNames[], we label all API
// records with a generic "HipApi" tag. Once the name table is available,
// dlsym it here and use it for per-API labeling.
static const char* api_name_for(uint32_t /*api_id*/) {
    return "HipApi";
}

// ============================================================
// Probe libamdhip64.so and resolve the 5 profiler symbols.
// Returns true if all symbols are available (feature is ready).
// Uses RTLD_NOLOAD so we do not force-load HIP if the app never used it.
// ============================================================
bool hip_profiler_probe() {
    if (g_api_ready) return true;

    // RTLD_NOLOAD: only succeed if libamdhip64.so is already resident.
    // If the app didn't use HIP, there are no records to drain anyway.
    g_hip_handle = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!g_hip_handle) {
        // Try without .so suffix variants. ROCm installs may symlink.
        g_hip_handle = dlopen("libamdhip64.so.7", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (!g_hip_handle) {
        g_hip_handle = dlopen("libamdhip64.so.6", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (!g_hip_handle) {
        fprintf(stderr, "rtl[hip]: libamdhip64.so not loaded — app did not use HIP, nothing to drain\n");
        return false;
    }

    // Clear dlerror before resolving.
    dlerror();
    g_fn_enable = reinterpret_cast<hipClrProfilerEnable_fn>(
        dlsym(g_hip_handle, "hipClrProfilerEnable"));
    g_fn_disable = reinterpret_cast<hipClrProfilerDisable_fn>(
        dlsym(g_hip_handle, "hipClrProfilerDisable"));
    g_fn_get_records = reinterpret_cast<hipClrProfilerGetRecords_fn>(
        dlsym(g_hip_handle, "hipClrProfilerGetRecords"));
    g_fn_reset = reinterpret_cast<hipClrProfilerReset_fn>(
        dlsym(g_hip_handle, "hipClrProfilerReset"));

    if (!g_fn_enable || !g_fn_disable || !g_fn_get_records || !g_fn_reset) {
        fprintf(stderr,
                "rtl[hip]: hipClrProfiler API not available in this ROCm build\n"
                "rtl[hip]:   enable=%p disable=%p get_records=%p reset=%p\n"
                "rtl[hip]: requires ROCm build with rocm-systems commit 5dc10a8 or later\n"
                "rtl[hip]: falling back to default mode would require a restart; this process will record nothing\n",
                (void*)g_fn_enable, (void*)g_fn_disable,
                (void*)g_fn_get_records, (void*)g_fn_reset);
        dlclose(g_hip_handle);
        g_hip_handle = nullptr;
        return false;
    }

    g_api_ready = true;
    fprintf(stderr, "rtl[hip]: hipClrProfiler API resolved — drain on exit enabled\n");
    return true;
}

// ============================================================
// Drain all accumulated HIP CLR profiler records into the trace database.
// Called from hsa_intercept::shutdown() (atexit handler).
//
// Ordering rationale:
//   - Disable() calls DrainAllDevices() internally, flushing in-flight GPU
//     work. After it returns all GPU activity records are finalized.
//   - At shutdown, app is exiting, no new HIP calls will be issued, so
//     GetRecords() is safe to call.
//   - Reset() frees the CLR-side buffer.
//
// The CLR profiler's HipClrProfilerFinalizer may still run later (static
// destructor order) and attempt to write JSON to GPU_CLR_PROFILE path, but
// we set that to /dev/null, so the write is a no-op.
// ============================================================
void hip_profiler_drain() {
    if (!hip_profiler_probe()) return;

    // Drain in-flight GPU work and stop collecting new records.
    int rc = g_fn_disable();
    if (rc != 0) {
        fprintf(stderr, "rtl[hip]: hipClrProfilerDisable returned %d\n", rc);
        // Continue anyway — GetRecords may still have usable data.
    }

    const HipClrApiRecord* records = nullptr;
    size_t count = 0;
    rc = g_fn_get_records(&records, &count);
    if (rc != 0 || !records) {
        fprintf(stderr, "rtl[hip]: hipClrProfilerGetRecords returned %d (count=%zu)\n",
                rc, count);
        return;
    }

    fprintf(stderr, "rtl[hip]: draining %zu records from HIP CLR profiler\n", count);

    // Thread / process identifiers for the HIP API rows.
    // CLR profiler provides a per-record thread_id (hash of std::thread::id)
    // which we use directly. The pid is the current process.
    const int pid = (int)getpid();

    auto& db = trace_db::get_trace_db();
    uint64_t n_kernel = 0, n_copy = 0, n_api = 0, n_skipped = 0;

    for (size_t i = 0; i < count; i++) {
        const HipClrApiRecord& rec = records[i];

        // Write the CPU-side HIP API record. correlation_id is the slot
        // index i — CLR profiler guarantees uniqueness within a process.
        const uint64_t corr_id = (uint64_t)i;
        const char* api_name = api_name_for(rec.api_id);

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

        // Write the GPU-side activity record if present.
        if (!rec.has_gpu_activity) continue;
        const HipClrGpuActivity& gpu = rec.gpu;
        if (gpu.end_ns <= gpu.begin_ns) {
            n_skipped++;
            continue;
        }

        switch (gpu.op) {
            case OP_DISPATCH: {
                const char* name = gpu.kernel_name ? gpu.kernel_name : "unknown_kernel";
                db.record_kernel(name, gpu.device_id, gpu.queue_id,
                                 gpu.begin_ns, gpu.end_ns, corr_id);
                n_kernel++;
                break;
            }
            case OP_COPY: {
                // CLR profiler gives us bytes but not src/dst device. Use
                // the activity's device_id as a stand-in for src_device and
                // -1 for dst_device (unknown direction).
                db.record_copy(gpu.device_id, -1, (size_t)gpu.bytes,
                               gpu.begin_ns, gpu.end_ns, corr_id);
                n_copy++;
                break;
            }
            case OP_BARRIER: {
                // Record as a synthetic kernel entry so it shows up in
                // Perfetto traces. Name it distinctly so summary tooling
                // can filter it out if needed.
                db.record_kernel("Barrier", gpu.device_id, gpu.queue_id,
                                 gpu.begin_ns, gpu.end_ns, corr_id);
                n_kernel++;
                break;
            }
            default:
                n_skipped++;
                break;
        }
    }

    fprintf(stderr, "rtl[hip]: drained api=%" PRIu64 " kernel=%" PRIu64
            " copy=%" PRIu64 " skipped=%" PRIu64 "\n",
            n_api, n_kernel, n_copy, n_skipped);

    // Free CLR-side buffer.
    g_fn_reset();
}

} // namespace hip_intercept
