// repro_passthrough_lib.cpp
//
// Minimal HSA_TOOLS_LIB that replaces hsa_queue_create with
// hsa_amd_queue_intercept_create + a pure passthrough callback.
//
// The callback does NOTHING except forward packets unchanged:
//   writer(in_packets, count)
//
// This is the simplest possible interception — no signal injection,
// no profiling, no data collection. Yet it causes a SEGFAULT during
// rapid hipGraph replay with 256+ kernels.
//
// Build:
//   g++ -shared -fPIC -o librepro_passthrough.so repro_passthrough_lib.cpp \
//       -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64
//
// Usage:
//   HSA_TOOLS_LIB=./librepro_passthrough.so ./repro_hipgraph_stress

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <atomic>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#ifndef AMD_INTERNAL_BUILD
#define AMD_INTERNAL_BUILD  // Use direct includes (not inc/ prefixed)
#endif
#include <hsa/hsa_api_trace.h>

// Saved original API tables
static CoreApiTable  g_orig_core;
static AmdExtTable   g_orig_ext;

static std::atomic<uint64_t> g_queues_created{0};
static std::atomic<uint64_t> g_callbacks_invoked{0};

// Pure passthrough callback — does absolutely nothing except forward packets.
static void passthrough_callback(const void* pkts, uint64_t pkt_count,
                                 uint64_t user_pkt_index, void* data,
                                 hsa_amd_queue_intercept_packet_writer writer) {
    g_callbacks_invoked.fetch_add(1, std::memory_order_relaxed);
    // Just forward unchanged — this is the minimal possible callback.
    writer(pkts, pkt_count);
}

// Replacement for hsa_queue_create that uses hsa_amd_queue_intercept_create.
static hsa_status_t my_hsa_queue_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
    void* data, uint32_t private_segment_size, uint32_t group_segment_size,
    hsa_queue_t** queue) {

    // hsa_amd_queue_intercept_create rejects UINT32_MAX segment sizes
    // (HIP passes UINT32_MAX to mean "use default"), so clamp to 0.
    uint32_t safe_priv = (private_segment_size == UINT32_MAX) ? 0 : private_segment_size;
    uint32_t safe_grp  = (group_segment_size == UINT32_MAX) ? 0 : group_segment_size;

    hsa_status_t status = g_orig_ext.hsa_amd_queue_intercept_create_fn(
        agent, size, type, callback, data,
        safe_priv, safe_grp, queue);

    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "repro: intercept_create failed (0x%x), falling back to plain queue\n",
                (unsigned)status);
        return g_orig_core.hsa_queue_create_fn(agent, size, type, callback, data,
                                                private_segment_size, group_segment_size, queue);
    }

    // Register the passthrough callback via saved API table
    g_orig_ext.hsa_amd_queue_intercept_register_fn(*queue, passthrough_callback, nullptr);

    uint64_t n = g_queues_created.fetch_add(1, std::memory_order_relaxed) + 1;
    fprintf(stderr, "repro: interceptible queue #%lu created (size=%u)\n",
            (unsigned long)n, size);

    return HSA_STATUS_SUCCESS;
}

extern "C" bool OnLoad(void* pTable,
                        uint64_t runtimeVersion,
                        uint64_t failedToolCount,
                        const char* const* pFailedToolNames) {
    fprintf(stderr, "repro: OnLoad (HSA runtime v%lu)\n", (unsigned long)runtimeVersion);

    auto* table = reinterpret_cast<HsaApiTable*>(pTable);

    // Save original tables
    g_orig_core = *table->core_;
    g_orig_ext  = *table->amd_ext_;

    // Replace queue creation only — nothing else
    table->core_->hsa_queue_create_fn = my_hsa_queue_create;

    fprintf(stderr, "repro: hsa_queue_create replaced with interceptible version\n");
    fprintf(stderr, "repro: callback does ONLY writer(pkts, count) — zero modification\n");
    return true;
}

extern "C" void OnUnload() {
    fprintf(stderr, "repro: OnUnload (queues created: %lu, callbacks invoked: %lu)\n",
            (unsigned long)g_queues_created.load(),
            (unsigned long)g_callbacks_invoked.load());
}
