/*
 * hsa_intercept.cpp — HSA_TOOLS_LIB based kernel dispatch profiling
 *
 * Uses HSA API table replacement (OnLoad) + hsa_amd_queue_intercept to
 * capture GPU kernel execution timestamps from AQL packets.
 *
 * Completion architecture:
 *   - Single worker thread processes completed dispatches from a queue
 *   - Original completion signals forwarded via HSA barrier packets
 *     (not manual signal manipulation)
 *   - Clean shutdown: worker joined before DB close
 *
 * Dependencies: libhsa-runtime64 only (part of ROCm runtime)
 */
#include "rpd_lite.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <hsa/amd_hsa_signal.h>

#include <cxxabi.h>

using namespace rpd_lite;

namespace hsa_intercept {

// Original HSA API tables
static CoreApiTable g_orig_core;
static AmdExtTable g_orig_ext;
static HsaApiTable* g_orig_table = nullptr;

// Kernel symbol map: code object handle -> name
static std::mutex g_symbol_mutex;
static std::unordered_map<uint64_t, std::string> g_symbols;

// Agent tracking (immutable after OnLoad)
static std::vector<hsa_agent_t> g_gpu_agents;
static std::mutex g_agent_mutex;

// Queue info: maps queue handle -> device index
struct QueueInfo {
    int device_id;
    uint64_t queue_handle;  // for stable queue identification
};
static std::mutex g_queue_mutex;
static std::unordered_map<uint64_t, QueueInfo> g_queue_map;  // keyed by hsa_queue_t pointer

static std::atomic<uint64_t> g_dispatch_id{0};

// ---- Completion worker ----
// Forward declarations
static std::string lookup_kernel_name(uint64_t kernel_object);

// Single thread processes all completed dispatches. Replaces thread-per-dispatch.

struct DispatchData {
    uint64_t dispatch_id;
    uint64_t correlation_id;
    uint64_t kernel_object;
    int device_id;
    uint64_t queue_id;
    hsa_signal_t profiling_signal;
};

static std::mutex g_work_mutex;
static std::condition_variable g_work_cv;
static std::deque<DispatchData*> g_work_queue;
static std::atomic<bool> g_shutdown{false};
static std::thread g_worker;

static void completion_worker() {
    while (true) {
        DispatchData* dd = nullptr;
        {
            std::unique_lock<std::mutex> lock(g_work_mutex);
            g_work_cv.wait(lock, [] {
                return !g_work_queue.empty() || g_shutdown.load(std::memory_order_acquire);
            });
            if (g_work_queue.empty()) {
                if (g_shutdown.load(std::memory_order_acquire)) break;
                continue;
            }
            dd = g_work_queue.front();
            g_work_queue.pop_front();
        }

        // Wait for kernel completion with bounded timeout.
        // HSA spec: timeout_hint is in nanoseconds (HSA Runtime Programmer's Reference 1.2+).
        // Uses 100ms timeout + poll loop so shutdown can interrupt.
        // Normal path: kernel completes in microseconds, first wait returns immediately.
        // Only during shutdown does the timeout loop matter (adds <=100ms exit latency).
        static constexpr uint64_t WAIT_TIMEOUT_NS = 100000000ULL;  // 100ms in nanoseconds
        hsa_signal_value_t wait_val;
        while (true) {
            wait_val = g_orig_core.hsa_signal_wait_scacquire_fn(
                dd->profiling_signal, HSA_SIGNAL_CONDITION_LT, 1,
                WAIT_TIMEOUT_NS, HSA_WAIT_STATE_BLOCKED);
            if (wait_val < 1) break;  // signal completed
            if (g_shutdown.load(std::memory_order_acquire)) {
                // Shutdown requested — abandon this dispatch, don't touch the signal
                delete dd;
                dd = nullptr;
                break;
            }
        }
        if (!dd) continue;  // was abandoned during shutdown

        // Read GPU timestamps
        hsa_amd_profiling_dispatch_time_t time{};
        hsa_status_t status = g_orig_ext.hsa_amd_profiling_get_dispatch_time_fn(
            g_gpu_agents[dd->device_id], dd->profiling_signal, &time);

        if (status != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "rpd_lite: warning: profiling_get_dispatch_time failed (status=%d) for dispatch %lu\n",
                    (int)status, dd->dispatch_id);
        } else if (time.end > time.start) {
            std::string name = lookup_kernel_name(dd->kernel_object);
            get_trace_db().record_kernel(name.c_str(), dd->device_id, dd->queue_id,
                                          time.start, time.end, dd->correlation_id);
        }

        // Do NOT destroy the signal — we don't own it (observe-only mode)
        delete dd;
    }
}

// ---- Kernel name lookup from code objects ----

static std::string lookup_kernel_name(uint64_t kernel_object) {
    std::lock_guard<std::mutex> lock(g_symbol_mutex);
    auto it = g_symbols.find(kernel_object);
    if (it != g_symbols.end()) {
        return it->second;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "kernel_0x%lx", kernel_object);
    return buf;
}

static hsa_status_t symbol_iterate_cb(hsa_executable_t exec,
                                       hsa_executable_symbol_t symbol,
                                       void* data) {
    hsa_symbol_kind_t kind;
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (kind != HSA_SYMBOL_KIND_KERNEL) return HSA_STATUS_SUCCESS;

    uint64_t handle;
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &handle) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    uint32_t name_len;
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_len) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    std::string name(name_len, '\0');
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &name[0]) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    std::lock_guard<std::mutex> lock(g_symbol_mutex);
    g_symbols[handle] = name;

    return HSA_STATUS_SUCCESS;
}

// Barrier/signal queue code removed — observe-only profiling doesn't need it.
// We pass packets through unmodified and read timestamps from original signals.

// ---- Queue intercept callback ----

static void queue_intercept_cb(const void* in_packets, uint64_t count,
                                uint64_t user_que_idx, void* data,
                                hsa_amd_queue_intercept_packet_writer writer) {
    if (g_shutdown.load(std::memory_order_acquire)) {
        writer(in_packets, count);
        return;
    }

    // Only handle single packets
    if (count != 1) {
        writer(in_packets, count);
        return;
    }

    const hsa_kernel_dispatch_packet_t* pkt =
        reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(in_packets);

    // Only intercept kernel dispatch packets
    uint8_t type = ((pkt->header >> HSA_PACKET_HEADER_TYPE) & 0xFF);
    if (type != HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        writer(in_packets, count);
        return;
    }

    // No completion signal → can't profile
    if (pkt->completion_signal.handle == 0) {
        writer(in_packets, count);
        return;
    }

    // Look up queue info
    QueueInfo* qi = static_cast<QueueInfo*>(data);
    if (!qi) {
        writer(in_packets, count);
        return;
    }

    // Observe-only profiling: pass packet through UNMODIFIED.
    // No signal replacement, no barrier packets, no extra queues.
    // We read GPU timestamps from the original signal after completion.
    writer(in_packets, count);

    // Build dispatch data — use the ORIGINAL signal for timestamp readback
    auto* dd = new DispatchData;
    dd->dispatch_id = g_dispatch_id.fetch_add(1, std::memory_order_relaxed);
    dd->correlation_id = next_correlation_id();
    dd->kernel_object = pkt->kernel_object;
    dd->device_id = qi->device_id;
    dd->queue_id = qi->queue_handle;
    dd->profiling_signal = pkt->completion_signal;  // observe, don't own

    // Enqueue for completion worker (non-blocking)
    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        g_work_queue.push_back(dd);
    }
    g_work_cv.notify_one();
}

// ---- HSA API table replacement ----

static hsa_status_t my_hsa_queue_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*),
    void* data, uint32_t private_segment_size,
    uint32_t group_segment_size, hsa_queue_t** queue) {

    // Create interceptible queue
    hsa_status_t status = g_orig_ext.hsa_amd_queue_intercept_create_fn(
        agent, size, type, callback, data,
        private_segment_size, group_segment_size, queue);

    if (status != HSA_STATUS_SUCCESS) return status;

    // Enable profiling on this queue
    hsa_status_t prof_status = g_orig_ext.hsa_amd_profiling_set_profiler_enabled_fn(*queue, true);
    if (prof_status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "rpd_lite: warning: failed to enable profiling on queue (status=%d)\n", (int)prof_status);
    }

    // Build queue info (no extra signal queue — observe-only profiling)
    auto* qi = new QueueInfo;
    qi->device_id = 0;
    qi->queue_handle = (uint64_t)(*queue);

    {
        std::lock_guard<std::mutex> lock(g_agent_mutex);
        for (size_t i = 0; i < g_gpu_agents.size(); i++) {
            if (g_gpu_agents[i].handle == agent.handle) {
                qi->device_id = (int)i;
                break;
            }
        }
    }

    // Track for cleanup
    {
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        g_queue_map[(uint64_t)(*queue)] = *qi;
    }

    // Register intercept callback with QueueInfo as userdata
    g_orig_ext.hsa_amd_queue_intercept_register_fn(*queue, queue_intercept_cb, qi);

    return HSA_STATUS_SUCCESS;
}

static hsa_status_t my_hsa_executable_freeze(hsa_executable_t executable, const char* options) {
    hsa_status_t status = g_orig_core.hsa_executable_freeze_fn(executable, options);
    if (status == HSA_STATUS_SUCCESS) {
        hsa_executable_iterate_symbols(executable, symbol_iterate_cb, nullptr);
    }
    return status;
}

static hsa_status_t agent_iterate_cb(hsa_agent_t agent, void* data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        std::lock_guard<std::mutex> lock(g_agent_mutex);
        g_gpu_agents.push_back(agent);
        g_orig_ext.hsa_amd_profiling_async_copy_enable_fn(true);
    }
    return HSA_STATUS_SUCCESS;
}

static void shutdown() {
    static std::atomic<bool> shutdown_done{false};
    if (shutdown_done.exchange(true)) return;  // prevent double shutdown

    // Signal worker to stop — worker checks g_shutdown in its wait loop
    g_shutdown.store(true, std::memory_order_release);
    g_work_cv.notify_all();

    // Join worker thread (will exit within 100ms due to timeout-based wait)
    if (g_worker.joinable()) {
        g_worker.join();
    }

    // Drain any remaining items in work queue (abandoned during shutdown)
    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        while (!g_work_queue.empty()) {
            delete g_work_queue.front();
            g_work_queue.pop_front();
        }
    }

    // Flush and close trace DB
    rpd_lite::get_trace_db().flush();
    rpd_lite::get_trace_db().close();

    // Clean up queue info (fixes memory leak from issue #5)
    {
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        g_queue_map.clear();
    }
}

} // namespace hsa_intercept

// ---- Entry points for HSA_TOOLS_LIB ----

extern "C" bool OnLoad(void* pTable,
                        uint64_t runtimeVersion,
                        uint64_t failedToolCount,
                        const char* const* pFailedToolNames) {
    fprintf(stderr, "rpd_lite: loading (HSA runtime v%lu)\n", runtimeVersion);

    using namespace hsa_intercept;

    // Ensure trace database is open
    (void)rpd_lite::get_trace_db();

    // Save original API tables
    HsaApiTable* table = reinterpret_cast<HsaApiTable*>(pTable);
    g_orig_table = table;
    g_orig_core = *table->core_;
    g_orig_ext = *table->amd_ext_;

    // Replace queue creation and executable freeze
    table->core_->hsa_queue_create_fn = my_hsa_queue_create;
    table->core_->hsa_executable_freeze_fn = my_hsa_executable_freeze;

    // Note: host timestamps (CLOCK_MONOTONIC) and GPU timestamps
    // (hsa_amd_profiling_get_dispatch_time) are in the same domain on ROCm/Linux
    // (both derive from TSC). This matches the original rtg_tracer approach.

    // Discover GPU agents (immutable after this point)
    hsa_iterate_agents(agent_iterate_cb, nullptr);
    fprintf(stderr, "rpd_lite: found %zu GPU agent(s)\n", g_gpu_agents.size());

    // Start completion worker thread
    g_worker = std::thread(completion_worker);
    fprintf(stderr, "rpd_lite: completion worker started\n");

    std::atexit(shutdown);

    return true;
}

extern "C" void OnUnload() {
    fprintf(stderr, "rpd_lite: unloading\n");
    hsa_intercept::shutdown();
}
