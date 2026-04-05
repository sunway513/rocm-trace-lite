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
    hsa_queue_t* signal_queue;  // per-queue signal queue for barrier forwarding
};
static std::mutex g_queue_mutex;
static std::unordered_map<uint64_t, QueueInfo> g_queue_map;  // keyed by hsa_queue_t pointer

static std::atomic<uint64_t> g_dispatch_id{0};

// ---- Completion worker ----
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
                return !g_work_queue.empty() || g_shutdown.load(std::memory_order_relaxed);
            });
            if (g_work_queue.empty()) {
                if (g_shutdown.load(std::memory_order_relaxed)) break;
                continue;
            }
            dd = g_work_queue.front();
            g_work_queue.pop_front();
        }

        // Blocking wait for kernel completion
        g_orig_core.hsa_signal_wait_scacquire_fn(
            dd->profiling_signal, HSA_SIGNAL_CONDITION_LT, 1,
            UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

        // Read GPU timestamps
        hsa_amd_profiling_dispatch_time_t time{};
        hsa_status_t status = g_orig_ext.hsa_amd_profiling_get_dispatch_time_fn(
            g_gpu_agents[dd->device_id], dd->profiling_signal, &time);

        if (status == HSA_STATUS_SUCCESS && time.end > time.start) {
            std::string name = lookup_kernel_name(dd->kernel_object);
            get_trace_db().record_kernel(name.c_str(), dd->device_id, dd->queue_id,
                                          time.start, time.end, dd->correlation_id);
        }

        // Destroy our profiling signal
        g_orig_core.hsa_signal_destroy_fn(dd->profiling_signal);
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

// ---- Signal queue for barrier-based completion forwarding ----
// Instead of manually decrementing the original completion signal,
// we submit a barrier packet to a signal queue that depends on our
// profiling signal and forwards completion to the original signal.
// This matches the proven rtg_tracer pattern.

static const uint16_t kBarrierHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const uint16_t kInvalidHeader =
    (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static void submit_barrier_to_forward_signal(hsa_queue_t* signal_queue,
                                              hsa_signal_t dep_signal,
                                              hsa_signal_t completion_signal) {
    if (!signal_queue || completion_signal.handle == 0) return;

    const uint32_t queue_mask = signal_queue->size - 1;
    uint64_t index = g_orig_core.hsa_queue_add_write_index_screlease_fn(signal_queue, 1);

    // Wait if queue is full
    while ((index - g_orig_core.hsa_queue_load_read_index_scacquire_fn(signal_queue)) >= queue_mask) {
        sched_yield();
    }

    hsa_barrier_and_packet_t* barrier =
        &((hsa_barrier_and_packet_t*)(signal_queue->base_address))[index & queue_mask];

    // Write packet with invalid header first (HSA convention)
    memset(barrier, 0, sizeof(*barrier));
    barrier->header = kInvalidHeader;
    barrier->completion_signal = completion_signal;
    barrier->dep_signal[0] = dep_signal;

    // Atomically set header to valid barrier (makes packet visible to HW)
    __atomic_store_n(&barrier->header, kBarrierHeader, __ATOMIC_RELEASE);
    g_orig_core.hsa_signal_store_relaxed_fn(signal_queue->doorbell_signal, index);
}

// ---- Queue intercept callback ----

static void queue_intercept_cb(const void* in_packets, uint64_t count,
                                uint64_t user_que_idx, void* data,
                                hsa_amd_queue_intercept_packet_writer writer) {
    if (g_shutdown.load(std::memory_order_relaxed)) {
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

    // Look up queue info
    QueueInfo* qi = static_cast<QueueInfo*>(data);
    if (!qi) {
        writer(in_packets, count);
        return;
    }

    // Create a profiling signal
    hsa_signal_t prof_signal;
    hsa_status_t status = g_orig_core.hsa_signal_create_fn(1, 0, nullptr, &prof_signal);
    if (status != HSA_STATUS_SUCCESS) {
        writer(in_packets, count);
        return;
    }

    // Build dispatch data for the completion worker
    auto* dd = new DispatchData;
    dd->dispatch_id = g_dispatch_id.fetch_add(1, std::memory_order_relaxed);
    dd->correlation_id = next_correlation_id();
    dd->kernel_object = pkt->kernel_object;
    dd->device_id = qi->device_id;
    dd->queue_id = qi->queue_handle;
    dd->profiling_signal = prof_signal;

    // Rewrite packet: replace completion signal with our profiling signal
    hsa_kernel_dispatch_packet_t modified = *pkt;
    modified.completion_signal = prof_signal;

    // Submit modified packet to GPU
    writer(&modified, 1);

    // Forward original completion via barrier packet (safe, proven pattern)
    if (pkt->completion_signal.handle != 0) {
        submit_barrier_to_forward_signal(qi->signal_queue, prof_signal, pkt->completion_signal);
    }

    // Enqueue for completion worker (non-blocking)
    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        g_work_queue.push_back(dd);
    }
    g_work_cv.notify_one();
}

// ---- HSA API table replacement ----

static hsa_queue_t* create_signal_queue(hsa_agent_t agent) {
    // Create a small queue dedicated to barrier packets for signal forwarding
    hsa_queue_t* queue = nullptr;
    hsa_status_t status = g_orig_core.hsa_queue_create_fn(
        agent, 128, HSA_QUEUE_TYPE_SINGLE,
        nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue);
    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "rpd_lite: warning: failed to create signal queue\n");
        return nullptr;
    }
    return queue;
}

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
    g_orig_ext.hsa_amd_profiling_set_profiler_enabled_fn(*queue, true);

    // Build queue info
    auto* qi = new QueueInfo;
    qi->device_id = 0;
    qi->queue_handle = (uint64_t)(*queue);
    qi->signal_queue = create_signal_queue(agent);

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
    // Signal worker to stop
    g_shutdown.store(true, std::memory_order_release);
    g_work_cv.notify_all();

    // Join worker thread
    if (g_worker.joinable()) {
        g_worker.join();
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
