/*
 * hsa_intercept.cpp — HSA_TOOLS_LIB based kernel dispatch profiling
 *
 * Uses HSA API table replacement (OnLoad) + hsa_amd_queue_intercept to
 * capture GPU kernel execution timestamps from AQL packets.
 *
 * Dependencies: libhsa-runtime64 only (part of ROCm runtime)
 */
#include "rpd_lite.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
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

// Agent tracking
static std::vector<hsa_agent_t> g_gpu_agents;
static std::mutex g_agent_mutex;

// Signal pool for profiling
static std::atomic<uint64_t> g_dispatch_id{0};

static std::string demangle(const char* name) {
    if (!name) return "<unknown>";
    int status;
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (demangled) {
        std::string result(demangled);
        free(demangled);
        return result;
    }
    return name;
}

// ---- Kernel name lookup from code objects ----

static hsa_status_t symbol_iterate_cb(hsa_executable_t exec,
                                       hsa_executable_symbol_t symbol,
                                       void* data) {
    hsa_symbol_kind_t kind;
    hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind);
    if (kind != HSA_SYMBOL_KIND_KERNEL) return HSA_STATUS_SUCCESS;

    uint64_t handle;
    hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &handle);

    uint32_t name_len;
    hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_len);

    std::string name(name_len, '\0');
    hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &name[0]);

    std::lock_guard<std::mutex> lock(g_symbol_mutex);
    g_symbols[handle] = name;

    return HSA_STATUS_SUCCESS;
}

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

// ---- Queue intercept callback ----
// This fires for every AQL packet submitted to a GPU queue.

struct DispatchData {
    uint64_t dispatch_id;
    uint64_t correlation_id;
    uint64_t kernel_object;
    int device_id;
    uint64_t queue_id;
    hsa_signal_t original_signal;
    hsa_signal_t profiling_signal;
};

static void signal_completion_handler(DispatchData* dd) {
    // Wait for the kernel to complete
    hsa_signal_value_t val = g_orig_core.hsa_signal_wait_scacquire_fn(
        dd->profiling_signal, HSA_SIGNAL_CONDITION_LT, 1,
        UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Read timestamps
    hsa_amd_profiling_dispatch_time_t time;
    g_orig_ext.hsa_amd_profiling_get_dispatch_time_fn(
        g_gpu_agents[dd->device_id], dd->profiling_signal, &time);

    // Record kernel execution
    std::string name = lookup_kernel_name(dd->kernel_object);
    get_trace_db().record_kernel(name.c_str(), dd->device_id, dd->queue_id,
                                  time.start, time.end, dd->correlation_id);

    // Signal the original completion signal if present
    if (dd->original_signal.handle != 0) {
        hsa_signal_value_t orig_val = g_orig_core.hsa_signal_load_relaxed_fn(dd->original_signal);
        g_orig_core.hsa_signal_store_screlease_fn(dd->original_signal, orig_val - 1);
    }

    g_orig_core.hsa_signal_destroy_fn(dd->profiling_signal);
    delete dd;
}

static void queue_intercept_cb(const void* in_packets, uint64_t count,
                                uint64_t user_que_idx, void* data,
                                hsa_amd_queue_intercept_packet_writer writer) {
    // We only handle single packets
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

    // Create a profiling signal
    hsa_signal_t prof_signal;
    hsa_status_t status = g_orig_core.hsa_signal_create_fn(1, 0, nullptr, &prof_signal);
    if (status != HSA_STATUS_SUCCESS) {
        writer(in_packets, count);
        return;
    }

    // Allocate dispatch data
    auto* dd = new DispatchData;
    dd->dispatch_id = g_dispatch_id.fetch_add(1);
    dd->correlation_id = next_correlation_id();
    dd->kernel_object = pkt->kernel_object;
    dd->original_signal = pkt->completion_signal;
    dd->profiling_signal = prof_signal;
    dd->queue_id = user_que_idx;

    // Determine device_id from queue data pointer
    // The 'data' parameter is passed from queue_create intercept
    dd->device_id = data ? *(int*)data : 0;

    // Rewrite the packet to use our profiling signal
    hsa_kernel_dispatch_packet_t modified = *pkt;
    modified.completion_signal = prof_signal;

    // Submit modified packet
    writer(&modified, 1);

    // Launch async wait thread for completion
    std::thread(signal_completion_handler, dd).detach();
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

    if (status == HSA_STATUS_SUCCESS) {
        // Enable profiling on this queue
        g_orig_ext.hsa_amd_profiling_set_profiler_enabled_fn(*queue, true);

        // Determine device index
        int* dev_idx = new int(0);
        {
            std::lock_guard<std::mutex> lock(g_agent_mutex);
            for (size_t i = 0; i < g_gpu_agents.size(); i++) {
                if (g_gpu_agents[i].handle == agent.handle) {
                    *dev_idx = (int)i;
                    break;
                }
            }
        }

        // Register our intercept callback
        g_orig_ext.hsa_amd_queue_intercept_register_fn(
            *queue, queue_intercept_cb, dev_idx);
    }

    return status;
}

static hsa_status_t my_hsa_executable_freeze(hsa_executable_t executable, const char* options) {
    hsa_status_t status = g_orig_core.hsa_executable_freeze_fn(executable, options);
    if (status == HSA_STATUS_SUCCESS) {
        // Enumerate symbols from newly loaded code objects
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

        // Enable async copy profiling
        g_orig_ext.hsa_amd_profiling_async_copy_enable_fn(true);
    }
    return HSA_STATUS_SUCCESS;
}

} // namespace hsa_intercept

// ---- Entry points for HSA_TOOLS_LIB ----

extern "C" bool OnLoad(void* pTable,
                        uint64_t runtimeVersion,
                        uint64_t failedToolCount,
                        const char* const* pFailedToolNames) {
    fprintf(stderr, "rpd_lite: loading (HSA runtime v%lu)\n", runtimeVersion);

    using namespace hsa_intercept;

    // Ensure trace database is open (lazy init handles dedup)
    (void)rpd_lite::get_trace_db();

    // Save original API tables
    HsaApiTable* table = reinterpret_cast<HsaApiTable*>(pTable);
    g_orig_table = table;
    g_orig_core = *table->core_;
    g_orig_ext = *table->amd_ext_;

    // Replace queue creation with intercepting version
    table->core_->hsa_queue_create_fn = my_hsa_queue_create;

    // Replace executable_freeze to capture kernel symbols
    table->core_->hsa_executable_freeze_fn = my_hsa_executable_freeze;

    // Discover GPU agents
    hsa_iterate_agents(agent_iterate_cb, nullptr);

    fprintf(stderr, "rpd_lite: found %zu GPU agent(s)\n", g_gpu_agents.size());

    // atexit is registered by lazy_init_db, no need to register again

    return true;
}

extern "C" void OnUnload() {
    fprintf(stderr, "rpd_lite: unloading\n");
    rpd_lite::get_trace_db().flush();
    rpd_lite::get_trace_db().close();
}
