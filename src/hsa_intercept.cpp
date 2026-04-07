/*
 * hsa_intercept.cpp — HSA_TOOLS_LIB based kernel dispatch profiling
 *
 * Uses HSA API table replacement (OnLoad) + hsa_amd_queue_intercept to
 * capture GPU kernel execution timestamps from AQL packets.
 *
 * Profiling architecture:
 *   - Inject profiling signals into kernel dispatch packets (HIP does not
 *     set completion_signal on most dispatches, so observe-only doesn't work)
 *   - Single worker thread processes completed dispatches from a queue
 *   - Original completion signals forwarded after profiling
 *   - Signal pool avoids per-dispatch hsa_signal_create overhead
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
#include <unistd.h>

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

// ---- Profiling signal pool ----
// Reuse HSA signals to avoid per-dispatch creation overhead.
static std::mutex g_pool_mutex;
static std::vector<hsa_signal_t> g_signal_pool;
static constexpr size_t SIGNAL_POOL_MAX = 4096;

static hsa_signal_t acquire_signal() {
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        if (!g_signal_pool.empty()) {
            hsa_signal_t sig = g_signal_pool.back();
            g_signal_pool.pop_back();
            g_orig_core.hsa_signal_store_relaxed_fn(sig, 1);
            return sig;
        }
    }
    hsa_signal_t sig;
    hsa_status_t st = g_orig_core.hsa_signal_create_fn(1, 0, nullptr, &sig);
    if (st != HSA_STATUS_SUCCESS) {
        sig.handle = 0;
    }
    return sig;
}

static void release_signal(hsa_signal_t sig) {
    if (sig.handle == 0) return;
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (g_signal_pool.size() < SIGNAL_POOL_MAX) {
        g_signal_pool.push_back(sig);
    } else {
        g_orig_core.hsa_signal_destroy_fn(sig);
    }
}

// ---- Diagnostic counters ----
static std::atomic<uint64_t> g_total_intercepts{0};
static std::atomic<uint64_t> g_drop_shutdown{0};
static std::atomic<uint64_t> g_drop_not_kernel{0};
static std::atomic<uint64_t> g_drop_no_qi{0};
static std::atomic<uint64_t> g_drop_sig_fail{0};     // signal pool exhausted
static std::atomic<uint64_t> g_drop_ts_fail{0};      // profiling_get_dispatch_time failed
static std::atomic<uint64_t> g_drop_ts_invalid{0};   // end <= start
static std::atomic<uint64_t> g_injected{0};           // signals injected
static std::atomic<uint64_t> g_injected_fallback{0};  // via kernel_object fallback
static std::atomic<uint64_t> g_recorded_ok{0};

// ---- Completion worker ----
static std::string lookup_kernel_name(uint64_t kernel_object);

struct DispatchData {
    uint64_t dispatch_id;
    uint64_t correlation_id;
    uint64_t kernel_object;
    int device_id;
    uint64_t queue_id;
    hsa_signal_t profiling_signal;  // injected signal (we own this)
    hsa_signal_t original_signal;   // original from packet (may be null)
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
        static constexpr uint64_t WAIT_TIMEOUT_NS = 100000000ULL;  // 100ms
        hsa_signal_value_t wait_val;
        bool abandoned = false;
        while (true) {
            wait_val = g_orig_core.hsa_signal_wait_scacquire_fn(
                dd->profiling_signal, HSA_SIGNAL_CONDITION_LT, 1,
                WAIT_TIMEOUT_NS, HSA_WAIT_STATE_BLOCKED);
            if (wait_val < 1) break;  // signal completed
            if (g_shutdown.load(std::memory_order_acquire)) {
                abandoned = true;
                break;
            }
        }

        if (abandoned) {
            // Forward original signal even during shutdown to avoid hangs.
            // Use subtract (decrement by 1) to match HSA packet completion semantics.
            if (dd->original_signal.handle != 0) {
                g_orig_core.hsa_signal_subtract_screlease_fn(dd->original_signal, 1);
            }
            release_signal(dd->profiling_signal);
            delete dd;
            continue;
        }

        // Read GPU timestamps from injected profiling signal
        hsa_amd_profiling_dispatch_time_t time{};
        if (dd->device_id < 0 || (size_t)dd->device_id >= g_gpu_agents.size()) {
            g_drop_ts_fail.fetch_add(1, std::memory_order_relaxed);
            if (dd->original_signal.handle != 0) {
                g_orig_core.hsa_signal_subtract_screlease_fn(dd->original_signal, 1);
            }
            release_signal(dd->profiling_signal);
            delete dd;
            continue;
        }
        hsa_status_t status = g_orig_ext.hsa_amd_profiling_get_dispatch_time_fn(
            g_gpu_agents[dd->device_id], dd->profiling_signal, &time);

        if (status != HSA_STATUS_SUCCESS) {
            g_drop_ts_fail.fetch_add(1, std::memory_order_relaxed);
        } else if (time.end <= time.start) {
            g_drop_ts_invalid.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::string name = lookup_kernel_name(dd->kernel_object);
            get_trace_db().record_kernel(name.c_str(), dd->device_id, dd->queue_id,
                                          time.start, time.end, dd->correlation_id);
            g_recorded_ok.fetch_add(1, std::memory_order_relaxed);
        }

        // Forward completion to original signal (if app set one).
        // HSA spec: packet completion decrements the signal by 1.
        // Use subtract_screlease to preserve original semantics exactly.
        if (dd->original_signal.handle != 0) {
            g_orig_core.hsa_signal_subtract_screlease_fn(dd->original_signal, 1);
        }

        // Return profiling signal to pool
        release_signal(dd->profiling_signal);
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
    if (name_len == 0) return HSA_STATUS_SUCCESS;

    std::string name(name_len, '\0');
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &name[0]) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    // Strip trailing ".kd" suffix (HSA kernel descriptor marker)
    if (name.size() > 3 && name.compare(name.size() - 3, 3, ".kd") == 0) {
        name.resize(name.size() - 3);
    }

    // Demangle C++ mangled symbols (start with "_Z") for readability.
    // Non-C++ names (Tensile Cijk, Triton, rocclr) pass through unchanged.
    if (name.size() >= 2 && name[0] == '_' && name[1] == 'Z') {
        int demangle_status = 0;
        char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &demangle_status);
        if (demangle_status == 0 && demangled) {
            // Extract readable short name from demangled C++ symbol.
            // "void ns::func<T>(args)" -> "ns::func<T>"
            std::string full(demangled);
            free(demangled);

            // Skip leading return type ("void ", "int ", etc.)
            size_t start = 0;
            auto first_colon = full.find("::");
            auto first_space = full.find(' ');
            if (first_space != std::string::npos && first_colon != std::string::npos
                && first_space < first_colon) {
                start = first_space + 1;
            }

            // Find the parameter list '(' — but skip "(anonymous namespace)"
            static constexpr const char* ANON_NS = "(anonymous namespace)";
            static constexpr size_t ANON_NS_LEN = 21;  // strlen("(anonymous namespace)")
            size_t end = std::string::npos;
            size_t pos = start;
            while (pos < full.size()) {
                pos = full.find('(', pos);
                if (pos == std::string::npos) break;
                if (full.compare(pos, ANON_NS_LEN, ANON_NS) == 0) {
                    pos += ANON_NS_LEN;
                    continue;
                }
                end = pos;
                break;
            }
            if (end == std::string::npos) end = full.size();

            name = full.substr(start, end - start);
        }
    }

    std::lock_guard<std::mutex> lock(g_symbol_mutex);
    g_symbols[handle] = name;

    return HSA_STATUS_SUCCESS;
}

// ---- Queue intercept callback ----

static void queue_intercept_cb(const void* in_packets, uint64_t count,
                                uint64_t user_que_idx, void* data,
                                hsa_amd_queue_intercept_packet_writer writer) {
    g_total_intercepts.fetch_add(1, std::memory_order_relaxed);

    if (g_shutdown.load(std::memory_order_acquire)) {
        g_drop_shutdown.fetch_add(count, std::memory_order_relaxed);
        writer(in_packets, count);
        return;
    }

    QueueInfo* qi = static_cast<QueueInfo*>(data);
    if (!qi) {
        g_drop_no_qi.fetch_add(count, std::memory_order_relaxed);
        writer(in_packets, count);
        return;
    }

    // Make a mutable copy of the packet buffer to inject profiling signals.
    // Stack buffer for common case (count <= 64), heap fallback for large batches.
    // HSA spec 1.2 section 2.8: all AQL packets are 64 bytes (fixed-width slots
    // in the ring buffer), regardless of packet type. Safe to step by fixed stride.
    const size_t pkt_size = sizeof(hsa_kernel_dispatch_packet_t);  // 64 bytes (AQL packet)
    const size_t buf_bytes = count * pkt_size;
    char stack_buf[64 * 64];
    std::vector<char> heap_buf;
    char* mod_buf;
    if (buf_bytes <= sizeof(stack_buf)) {
        mod_buf = stack_buf;
    } else {
        heap_buf.resize(buf_bytes);
        mod_buf = heap_buf.data();
    }
    memcpy(mod_buf, in_packets, buf_bytes);

    // Collect dispatch data to enqueue after writer().
    // Stack array for common case, vector fallback for large batches.
    DispatchData* pending_stack[64];
    std::vector<DispatchData*> pending_vec;
    DispatchData** dd_list;
    if (count <= 64) {
        dd_list = pending_stack;
    } else {
        pending_vec.resize(count);
        dd_list = pending_vec.data();
    }
    uint64_t dd_count = 0;

    for (uint64_t i = 0; i < count; i++) {
        hsa_kernel_dispatch_packet_t* pkt =
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(mod_buf + i * pkt_size);

        // Detect kernel dispatch packets.
        // Primary: header type == KERNEL_DISPATCH (standard HSA).
        // Fallback: some runtimes use type=0 (VENDOR_SPECIFIC) for kernel
        // dispatches. Validate multiple dispatch-specific fields to avoid
        // misclassifying non-dispatch packets.
        uint8_t type = ((pkt->header >> HSA_PACKET_HEADER_TYPE) & 0xFF);
        bool is_kernel = (type == HSA_PACKET_TYPE_KERNEL_DISPATCH);
        if (!is_kernel
            && pkt->kernel_object != 0
            && pkt->workgroup_size_x > 0
            && pkt->grid_size_x > 0) {
            is_kernel = true;
            g_injected_fallback.fetch_add(1, std::memory_order_relaxed);
        }
        if (!is_kernel) {
            g_drop_not_kernel.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Acquire a profiling signal from pool
        hsa_signal_t prof_sig = acquire_signal();
        if (prof_sig.handle == 0) {
            g_drop_sig_fail.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Save original signal and inject profiling signal
        hsa_signal_t orig_sig = pkt->completion_signal;
        pkt->completion_signal = prof_sig;

        auto* dd = new DispatchData;
        dd->dispatch_id = g_dispatch_id.fetch_add(1, std::memory_order_relaxed);
        dd->correlation_id = next_correlation_id();
        dd->kernel_object = pkt->kernel_object;
        dd->device_id = qi->device_id;
        dd->queue_id = qi->queue_handle;
        dd->profiling_signal = prof_sig;
        dd->original_signal = orig_sig;

        dd_list[dd_count++] = dd;
        g_injected.fetch_add(1, std::memory_order_relaxed);
    }

    // Submit (potentially modified) packets to hardware
    writer(mod_buf, count);

    // Enqueue all dispatch data for completion worker
    if (dd_count > 0) {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        for (uint64_t i = 0; i < dd_count; i++) {
            g_work_queue.push_back(dd_list[i]);
        }
        g_work_cv.notify_one();
    }
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
        fprintf(stderr, "rtl: warning: failed to enable profiling on queue (status=%d)\n", (int)prof_status);
    }

    // Build queue info
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

    // Print diagnostic counters
    fprintf(stderr, "\n=== rtl diagnostic (PID %d) ===\n", getpid());
    fprintf(stderr, "  intercept calls:     %lu\n", g_total_intercepts.load());
    fprintf(stderr, "  signals injected:    %lu\n", g_injected.load());
    if (g_injected_fallback.load() > 0) {
        fprintf(stderr, "  fallback detect:     %lu\n", g_injected_fallback.load());
    }
    fprintf(stderr, "  drop (shutdown):     %lu\n", g_drop_shutdown.load());
    fprintf(stderr, "  drop (not kernel):   %lu\n", g_drop_not_kernel.load());
    fprintf(stderr, "  drop (no qi):        %lu\n", g_drop_no_qi.load());
    fprintf(stderr, "  drop (sig pool):     %lu\n", g_drop_sig_fail.load());
    fprintf(stderr, "  drop (ts fail):      %lu\n", g_drop_ts_fail.load());
    fprintf(stderr, "  drop (ts invalid):   %lu\n", g_drop_ts_invalid.load());
    fprintf(stderr, "  recorded OK:         %lu\n", g_recorded_ok.load());
    fprintf(stderr, "====================================\n\n");

    // Signal worker to stop
    g_shutdown.store(true, std::memory_order_release);
    g_work_cv.notify_all();

    if (g_worker.joinable()) {
        g_worker.join();
    }

    // Drain remaining work queue
    {
        std::lock_guard<std::mutex> lock(g_work_mutex);
        while (!g_work_queue.empty()) {
            auto* dd = g_work_queue.front();
            g_work_queue.pop_front();
            // Forward original signals to avoid app hangs
            if (dd->original_signal.handle != 0) {
                g_orig_core.hsa_signal_subtract_screlease_fn(dd->original_signal, 1);
            }
            release_signal(dd->profiling_signal);
            delete dd;
        }
    }

    // Flush and close trace DB
    rpd_lite::get_trace_db().flush();
    rpd_lite::get_trace_db().close();

    // Destroy signal pool
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        for (auto& sig : g_signal_pool) {
            g_orig_core.hsa_signal_destroy_fn(sig);
        }
        g_signal_pool.clear();
    }

    // Clean up queue info
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
    fprintf(stderr, "rtl: loading (HSA runtime v%lu)\n", runtimeVersion);

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
    fprintf(stderr, "rtl: found %zu GPU agent(s)\n", g_gpu_agents.size());

    // Pre-allocate signal pool (256 initial for multi-GPU workloads)
    g_signal_pool.reserve(512);
    for (int i = 0; i < 256; i++) {
        hsa_signal_t sig;
        if (g_orig_core.hsa_signal_create_fn(1, 0, nullptr, &sig) == HSA_STATUS_SUCCESS) {
            g_signal_pool.push_back(sig);
        }
    }
    fprintf(stderr, "rtl: signal pool initialized (%zu signals)\n", g_signal_pool.size());

    // Start completion worker thread
    g_worker = std::thread(completion_worker);
    fprintf(stderr, "rtl: completion worker started\n");

    std::atexit(shutdown);

    return true;
}

extern "C" void OnUnload() {
    fprintf(stderr, "rtl: unloading\n");
    hsa_intercept::shutdown();
}
