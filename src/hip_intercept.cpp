/*
 * hip_intercept.cpp -- HIP CLR profiler integration (RTL_MODE=hip)
 *
 * Uses the HIP profiler extension API (hipProfiler*Ext) from CLR's built-in
 * profiling layer. Requires ROCm/rocm-systems PR #5215 (branch
 * amd/dev/gandryey/ROCM-1667-12) or later.
 *
 * Activation: GPU_CLR_PROFILE_OUTPUT=/dev/null triggers CLR's dispatch
 * wrappers during hip::init(). At shutdown, drain() calls Disable →
 * GetRecords to harvest all records (zero-copy chunked access).
 *
 * API v0.1.0 record layout:
 *   HipApiRecordExt (256B): api_name, TID, timestamps, stream,
 *     memory1/memory2/size (for alloc/copy ops), embedded HipGpuActivityExt
 *   HipGpuActivityExt (128B): op type, device/queue, GPU timestamps,
 *     kernel_name + grid/block dims (dispatch), src/dst addrs (copy),
 *     linked-list `next` for multi-op graph launches
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
// HIP CLR profiler types.
//
// Prefer the official header (shipped by German's CLR build in
// amd/dev/gandryey/ROCM-1667-12; path: hip/amd_detail/hip_profiler_ext.h).
// Fall back to local copies for builds against stock ROCm.
//
// Layout must stay in sync with ROCm/rocm-systems branch
// amd/dev/gandryey/ROCM-1667-12 (commit 8e637b7 or later).
// ============================================================
#if __has_include(<hip/amd_detail/hip_profiler_ext.h>)
#  include <hip/amd_detail/hip_profiler_ext.h>
#define RTL_HIP_PROFILER_EXT_TYPES_FROM_HEADER 1
// Map official enum names to our local names
static constexpr uint32_t OP_DISPATCH = HIP_OP_DISPATCH_EXT;
static constexpr uint32_t OP_COPY     = HIP_OP_COPY_EXT;
static constexpr uint32_t OP_BARRIER  = HIP_OP_BARRIER_EXT;
static constexpr uint32_t COPY_H2D       = HIP_COPY_KIND_H2D_EXT;
static constexpr uint32_t COPY_H2D_IMAGE = HIP_COPY_KIND_H2D_IMAGE_EXT;
static constexpr uint32_t COPY_D2H       = HIP_COPY_KIND_D2H_EXT;
static constexpr uint32_t COPY_D2H_IMAGE = HIP_COPY_KIND_D2H_IMAGE_EXT;
static constexpr uint32_t COPY_D2D       = HIP_COPY_KIND_D2D_EXT;
static constexpr uint32_t COPY_D2D_IMAGE = HIP_COPY_KIND_D2D_IMAGE_EXT;
#else

enum HipGpuOpExt : uint32_t {
    OP_DISPATCH = 0,
    OP_COPY     = 1,
    OP_BARRIER  = 2,
};

// SDMA copy direction (4-bit, stored in HipGpuActivityExt bitfield)
enum HipCopyKindExt : uint32_t {
    COPY_UNKNOWN         = 0,
    COPY_H2D             = 1,
    COPY_H2D_RECT        = 2,
    COPY_H2D_IMAGE       = 3,
    COPY_D2H             = 4,
    COPY_D2H_RECT        = 5,
    COPY_D2H_IMAGE       = 6,
    COPY_D2D             = 7,
    COPY_D2D_RECT        = 8,
    COPY_D2D_IMAGE       = 9,
    COPY_BUF_TO_IMAGE    = 10,
    COPY_IMAGE_TO_BUF    = 11,
    COPY_FILL            = 12,
};

static const char* copy_kind_name(uint32_t kind) {
    switch (kind) {
        case COPY_H2D:          return "H2D";
        case COPY_H2D_RECT:    return "H2D_Rect";
        case COPY_H2D_IMAGE:   return "H2D_Image";
        case COPY_D2H:          return "D2H";
        case COPY_D2H_RECT:    return "D2H_Rect";
        case COPY_D2H_IMAGE:   return "D2H_Image";
        case COPY_D2D:          return "D2D";
        case COPY_D2D_RECT:    return "D2D_Rect";
        case COPY_D2D_IMAGE:   return "D2D_Image";
        case COPY_BUF_TO_IMAGE: return "BufToImage";
        case COPY_IMAGE_TO_BUF: return "ImageToBuf";
        case COPY_FILL:         return "Fill";
        default:                return "Copy";
    }
}

// Returns true if copy_kind represents a PCIe SDMA transfer (H2D or D2H).
static bool is_sdma_copy(uint32_t kind) {
    return (kind >= COPY_H2D && kind <= COPY_H2D_IMAGE) ||
           (kind >= COPY_D2H && kind <= COPY_D2H_IMAGE);
}

struct HipGpuActivityExt {
    union {
        uint64_t _flags_u64;
        struct {
            uint64_t op        : 3;   // HipGpuOpExt
            uint64_t is_graph  : 1;
            uint64_t copy_kind : 4;   // HipCopyKindExt (valid when op==OP_COPY)
            uint64_t           : 8;
            uint64_t device_id : 16;
            uint64_t queue_id  : 16;
            uint64_t           : 16;
        };
    };
    uint64_t    begin_ns;
    uint64_t    end_ns;
    union {
        uint64_t    bytes;            // copy ops
        const char* kernel_name;      // dispatch ops (demangled, may be NULL)
    };
    uint32_t    gpu_op_count;
    uint32_t    _reserved_u32;
    const HipGpuActivityExt* next;    // linked list for multi-op graph launches
    union {
        struct { // op==OP_DISPATCH
            uint32_t       grid_x;
            uint32_t       grid_y;
            uint32_t       grid_z;
            uint32_t       block_x;
            uint32_t       block_y;
            uint32_t       block_z;
            const uint8_t* kernel_args;
            uint32_t       kernel_args_size;
            uint32_t       _reserved_dispatch;
        };
        struct { // op==OP_COPY
            const void*    src;
            const void*    dst;
            uint8_t        _reserved_copy[24];
        };
    };
    uint8_t     _pad1[40];
};
static_assert(sizeof(HipGpuActivityExt) == 128, "HipGpuActivityExt must be 128 bytes");

struct HipApiRecordExt {
    const char*  api_name;
    union {
        uint64_t _flags_u64;
        struct {
            uint64_t has_gpu_activity : 1;
            uint64_t                  : 63;
        };
    };
    uint64_t     thread_id;
    uint64_t     start_ns;
    uint64_t     end_ns;
    void*        stream;               // hipStream_t
    const HipGpuActivityExt* _spill_tail; // internal, do not use
    void*        memory1;              // dst/alloc ptr, or graphExec for hipGraphLaunch
    void*        memory2;              // src ptr for copies
    uint64_t     size;                 // bytes for allocs/copies
    uint8_t      _pad1[48];
    HipGpuActivityExt gpu;
};
static_assert(sizeof(HipApiRecordExt) == 256, "HipApiRecordExt must be 256 bytes");
#endif // !RTL_HIP_PROFILER_EXT_TYPES_FROM_HEADER

// v0.1.0 signatures (PR #5215): Enable/Disable take uint64_t* output param
using hipProfilerEnableExt_fn     = int (*)(uint64_t*);
using hipProfilerDisableExt_fn    = int (*)(uint64_t*);
using hipProfilerGetRecordsExt_fn = int (*)(const HipApiRecordExt* const** chunks,
                                            size_t* chunk_count,
                                            size_t* chunk_size,
                                            size_t* total_count);
// v0 had Reset; v0.1.0 removed it. Probe at runtime.
using hipProfilerResetExt_fn      = int (*)(void);

static void* g_hip_handle = nullptr;
static hipProfilerEnableExt_fn     g_fn_enable      = nullptr;
static hipProfilerDisableExt_fn    g_fn_disable     = nullptr;
static hipProfilerGetRecordsExt_fn g_fn_get_records = nullptr;
static hipProfilerResetExt_fn      g_fn_reset       = nullptr;  // optional (v0 only)
static bool g_api_ready = false;
static bool g_has_linked_list_api = false;  // v0.1.0 uses next linked list vs v0 gpu_ops array

static void write_gpu_op(trace_db::TraceDB& db,
                         const HipGpuActivityExt& gpu,
                         uint64_t corr_id,
                         uint64_t& n_kernel,
                         uint64_t& n_copy,
                         uint64_t& n_barrier,
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
            int src_dev = (int)gpu.device_id, dst_dev = -1;
            uint32_t ck = (uint32_t)gpu.copy_kind;
            if (ck >= COPY_H2D && ck <= COPY_H2D_IMAGE) {
                src_dev = -1; dst_dev = (int)gpu.device_id;
            } else if (ck >= COPY_D2H && ck <= COPY_D2H_IMAGE) {
                src_dev = (int)gpu.device_id; dst_dev = -1;
            } else if (ck >= COPY_D2D && ck <= COPY_D2D_IMAGE) {
                src_dev = (int)gpu.device_id; dst_dev = (int)gpu.device_id;
            }
            db.record_copy(src_dev, dst_dev, (size_t)gpu.bytes,
                           gpu.begin_ns, gpu.end_ns, corr_id);
            n_copy++;
            break;
        }
        case OP_BARRIER: {
            db.record_kernel("Barrier", (int)gpu.device_id, gpu.queue_id,
                             gpu.begin_ns, gpu.end_ns, corr_id);
            n_barrier++;
            break;
        }
        default:
            n_skipped++;
            break;
    }
}

bool hip_profiler_probe() {
    if (g_api_ready) return true;

    g_hip_handle = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!g_hip_handle)
        g_hip_handle = dlopen("libamdhip64.so.7", RTLD_LAZY | RTLD_NOLOAD);
    if (!g_hip_handle)
        g_hip_handle = dlopen("libamdhip64.so.6", RTLD_LAZY | RTLD_NOLOAD);
    if (!g_hip_handle) {
        fprintf(stderr, "rtl[hip]: libamdhip64.so not loaded, nothing to drain\n");
        return false;
    }

    dlerror();

    g_fn_enable = reinterpret_cast<hipProfilerEnableExt_fn>(
        dlsym(g_hip_handle, "hipProfilerEnableExt"));
    g_fn_disable = reinterpret_cast<hipProfilerDisableExt_fn>(
        dlsym(g_hip_handle, "hipProfilerDisableExt"));
    g_fn_get_records = reinterpret_cast<hipProfilerGetRecordsExt_fn>(
        dlsym(g_hip_handle, "hipProfilerGetRecordsExt"));
    // Reset only exists in v0 API; v0.1.0 removed it
    g_fn_reset = reinterpret_cast<hipProfilerResetExt_fn>(
        dlsym(g_hip_handle, "hipProfilerResetExt"));

    if (g_fn_enable && g_fn_disable && g_fn_get_records) {
        g_api_ready = true;
        // v0.1.0: no Reset, uses linked-list `next` for graph ops
        // v0: has Reset, uses contiguous `gpu_ops` array
        g_has_linked_list_api = (g_fn_reset == nullptr);
        uint64_t start_id = 0;
        int rc = g_fn_enable(&start_id);
        fprintf(stderr, "rtl[hip]: hipProfiler*Ext API %s resolved, "
                "Enable returned %d (start_id=%" PRIu64 ")\n",
                g_has_linked_list_api ? "v0.1.0" : "v0", rc, start_id);
        return true;
    }

    fprintf(stderr,
            "rtl[hip]: hipProfiler API not available in this ROCm build\n"
            "rtl[hip]:   enable=%p disable=%p get_records=%p\n"
            "rtl[hip]: requires CLR with ROCM-1667-12 profiler extension\n",
            (void*)g_fn_enable, (void*)g_fn_disable, (void*)g_fn_get_records);
    dlclose(g_hip_handle);
    g_hip_handle = nullptr;
    return false;
}

static void format_api_args(const HipApiRecordExt& rec,
                            const HipGpuActivityExt& gpu,
                            char* buf, size_t buf_sz) {
    buf[0] = '\0';
    if (gpu.op == OP_DISPATCH && gpu.grid_x > 0) {
        snprintf(buf, buf_sz, "grid=[%u,%u,%u] block=[%u,%u,%u]",
                 gpu.grid_x, gpu.grid_y, gpu.grid_z,
                 gpu.block_x, gpu.block_y, gpu.block_z);
    } else if (rec.memory1 || rec.memory2 || rec.size) {
        if (rec.memory1 && rec.memory2) {
            snprintf(buf, buf_sz, "dst=%p src=%p size=%zu",
                     rec.memory1, rec.memory2, (size_t)rec.size);
        } else if (rec.memory1) {
            snprintf(buf, buf_sz, "ptr=%p size=%zu",
                     rec.memory1, (size_t)rec.size);
        }
    }
}

void hip_profiler_drain() {
    if (!hip_profiler_probe()) return;

    uint64_t end_id = 0;
    int rc = g_fn_disable(&end_id);
    if (rc != 0) {
        fprintf(stderr, "rtl[hip]: hipProfilerDisableExt returned %d\n", rc);
    }

    const HipApiRecordExt* const* chunks = nullptr;
    size_t chunk_count = 0, chunk_size = 0, total_count = 0;
    rc = g_fn_get_records(&chunks, &chunk_count, &chunk_size, &total_count);
    if (rc != 0 || !chunks || chunk_count == 0) {
        fprintf(stderr, "rtl[hip]: hipProfilerGetRecordsExt returned %d "
                "(chunks=%zu, chunk_size=%zu, total=%zu)\n",
                rc, chunk_count, chunk_size, total_count);
        return;
    }

    fprintf(stderr, "rtl[hip]: draining %zu records (%zu chunks x %zu) from HIP profiler\n",
            total_count, chunk_count, chunk_size);

    const int pid = (int)getpid();
    auto& db = trace_db::get_trace_db();
    uint64_t n_kernel = 0, n_copy = 0, n_barrier = 0;
    uint64_t n_api = 0, n_graph_ops = 0, n_skipped = 0;
    uint64_t global_idx = 0;

    for (size_t c = 0; c < chunk_count; c++) {
        size_t remaining = total_count - c * chunk_size;
        size_t n = (remaining < chunk_size) ? remaining : chunk_size;

        for (size_t i = 0; i < n; i++, global_idx++) {
            const HipApiRecordExt& rec = chunks[c][i];
            const uint64_t corr_id = global_idx;
            const char* api_name = rec.api_name ? rec.api_name : "HipApi";

            // Build args string from new fields
            char args[256];
            format_api_args(rec, rec.gpu, args, sizeof(args));

            if (rec.end_ns > rec.start_ns) {
                db.record_hip_api(api_name, args,
                                  rec.start_ns,
                                  rec.end_ns - rec.start_ns,
                                  corr_id, pid, (int)rec.thread_id);
                n_api++;
            } else {
                n_skipped++;
            }

            if (rec.gpu.gpu_op_count == 0) continue;

            if (rec.gpu.gpu_op_count > 1 && rec.gpu.next) {
                if (g_has_linked_list_api) {
                    // v0.1.0: traverse linked list via `next` pointer
                    const HipGpuActivityExt* node = &rec.gpu;
                    while (node) {
                        write_gpu_op(db, *node, corr_id,
                                     n_kernel, n_copy, n_barrier, n_skipped);
                        n_graph_ops++;
                        node = node->next;
                    }
                } else {
                    // v0: contiguous array via `gpu_ops` pointer (same offset as `next`)
                    const HipGpuActivityExt* arr =
                        reinterpret_cast<const HipGpuActivityExt*>(rec.gpu.next);
                    for (uint32_t j = 0; j < rec.gpu.gpu_op_count; j++) {
                        write_gpu_op(db, arr[j], corr_id,
                                     n_kernel, n_copy, n_barrier, n_skipped);
                        n_graph_ops++;
                    }
                }
            } else {
                write_gpu_op(db, rec.gpu, corr_id,
                             n_kernel, n_copy, n_barrier, n_skipped);
            }
        }
    }

    fprintf(stderr, "rtl[hip]: drained api=%" PRIu64 " kernel=%" PRIu64
            " copy=%" PRIu64 " barrier=%" PRIu64
            " graph_ops=%" PRIu64 " skipped=%" PRIu64 "\n",
            n_api, n_kernel, n_copy, n_barrier, n_graph_ops, n_skipped);

    if (g_fn_reset) g_fn_reset();  // v0 only; v0.1.0 has no Reset
}

} // namespace hip_intercept
