/*
 * rocprofiler_backend.cpp — rocprofiler-sdk buffer tracing backend
 *
 * Uses rocprofiler-sdk's async buffer tracing API for kernel dispatch timing.
 * Buffer tracing delivers records on a background thread, avoiding deadlocks
 * during framework initialization and capturing CUDA graph replay dispatches.
 *
 * Entry point: rocprofiler_configure() — discovered by rocprofiler-register.
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 */

#include "trace_db.h"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <inttypes.h>
#include <mutex>
#include <string>
#include <unistd.h>
#include <unordered_map>

namespace {

rocprofiler_context_id_t g_ctx;
rocprofiler_buffer_id_t  g_buffer;

std::unordered_map<uint64_t, int> g_agent_device_map;
std::mutex                        g_agent_mutex;

std::unordered_map<uint64_t, std::string> g_kernel_names;
std::mutex                                g_kernel_mutex;

std::atomic<bool> g_shutdown_done{false};
std::atomic<uint64_t> g_recorded{0};

// ---- Kernel name demangling ----

std::string demangle_kernel_name(const char* raw) {
    if (!raw || !*raw) return "<unknown>";
    std::string name(raw);
    if (name.size() > 3 && name.compare(name.size() - 3, 3, ".kd") == 0)
        name.resize(name.size() - 3);
    if (name.size() >= 2 && name[0] == '_' && name[1] == 'Z') {
        int status = 0;
        char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
        if (status == 0 && demangled) {
            std::string full(demangled);
            free(demangled);
            size_t start = 0;
            auto fc = full.find("::");
            auto fs = full.find(' ');
            if (fs != std::string::npos && fc != std::string::npos && fs < fc)
                start = fs + 1;
            static constexpr const char* ANON_NS = "(anonymous namespace)";
            size_t end = std::string::npos, pos = start;
            while (pos < full.size()) {
                pos = full.find('(', pos);
                if (pos == std::string::npos) break;
                if (full.compare(pos, 21, ANON_NS) == 0) { pos += 21; continue; }
                end = pos; break;
            }
            if (end == std::string::npos) end = full.size();
            name = full.substr(start, end - start);
        }
    }
    return name;
}

const std::string& resolve_kernel_name(uint64_t kernel_id) {
    static const std::string unknown = "<unknown>";
    std::lock_guard<std::mutex> lock(g_kernel_mutex);
    auto it = g_kernel_names.find(kernel_id);
    return (it != g_kernel_names.end()) ? it->second : unknown;
}

int resolve_device_id(uint64_t agent_handle) {
    std::lock_guard<std::mutex> lock(g_agent_mutex);
    auto it = g_agent_device_map.find(agent_handle);
    return (it != g_agent_device_map.end()) ? it->second : -1;
}

// ---- Buffer tracing callback (runs on background thread) ----

void buffer_tracing_callback(rocprofiler_context_id_t,
                              rocprofiler_buffer_id_t,
                              rocprofiler_record_header_t** headers,
                              size_t                        num_headers,
                              void*,
                              uint64_t                      drop_count) {
    if (drop_count > 0)
        fprintf(stderr, "rtl: WARNING: %" PRIu64 " records dropped (buffer full)\n", drop_count);

    for (size_t i = 0; i < num_headers; ++i) {
        auto* hdr = headers[i];
        if (hdr->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
            hdr->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {

            auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                hdr->payload);

            const auto& info = record->dispatch_info;
            int device_id = resolve_device_id(info.agent_id.handle);
            const std::string& kname = resolve_kernel_name(info.kernel_id);

            char dispatch_info[128];
            snprintf(dispatch_info, sizeof(dispatch_info),
                     "wg=%u,%u,%u grid=%u,%u,%u",
                     info.workgroup_size.x, info.workgroup_size.y, info.workgroup_size.z,
                     info.grid_size.x, info.grid_size.y, info.grid_size.z);

            trace_db::get_trace_db().record_kernel(
                kname.c_str(),
                device_id,
                info.queue_id.handle,
                record->start_timestamp,
                record->end_timestamp,
                record->correlation_id.internal,
                dispatch_info,
                0);

            g_recorded.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// ---- Code object callback (synchronous — kernel symbol registration is safe) ----

void code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*,
                          void*) {
    if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if (record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) return;
    if (record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    auto* data = static_cast<
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(record.payload);
    if (!data || !data->kernel_name) return;

    std::string demangled = demangle_kernel_name(data->kernel_name);
    std::lock_guard<std::mutex> lock(g_kernel_mutex);
    g_kernel_names[data->kernel_id] = std::move(demangled);
}

// ---- Agent enumeration ----

rocprofiler_status_t agent_enumerate_cb(rocprofiler_agent_version_t,
                                        const void** agents,
                                        size_t       num_agents,
                                        void*) {
    int gpu_idx = 0;
    std::lock_guard<std::mutex> lock(g_agent_mutex);
    for (size_t i = 0; i < num_agents; ++i) {
        auto* agent = static_cast<const rocprofiler_agent_t*>(agents[i]);
        if (agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            g_agent_device_map[agent->id.handle] = gpu_idx++;
    }
    fprintf(stderr, "rtl: found %d GPU agent(s) [rocprofiler-sdk]\n", gpu_idx);
    return ROCPROFILER_STATUS_SUCCESS;
}

// ---- Shutdown ----

void shutdown() {
    bool expected = false;
    if (!g_shutdown_done.compare_exchange_strong(expected, true)) return;

    rocprofiler_flush_buffer(g_buffer);

    fprintf(stderr, "rtl: recorded %" PRIu64 " kernel dispatches (pid=%d)\n",
            g_recorded.load(), getpid());

    auto& db = trace_db::get_trace_db();
    db.flush();
    db.close();
    fprintf(stderr, "rtl: rocprofiler-sdk backend shutdown complete (pid=%d)\n", getpid());
}

// ---- Tool init / fini ----

int rtl_tool_init(rocprofiler_client_finalize_t, void*) {
    const char* output_env = std::getenv("RTL_OUTPUT");
    std::string db_path = output_env ? output_env : "rtl_trace_%p.db";
    auto pos = db_path.find("%p");
    if (pos != std::string::npos)
        db_path.replace(pos, 2, std::to_string(getpid()));

    if (!trace_db::get_trace_db().open(db_path)) {
        fprintf(stderr, "rtl: ERROR: failed to open trace DB: %s\n", db_path.c_str());
        return -1;
    }
    fprintf(stderr, "rtl: trace DB opened: %s\n", db_path.c_str());

    rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0, agent_enumerate_cb,
        sizeof(rocprofiler_agent_t), nullptr);

    auto status = rocprofiler_create_context(&g_ctx);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: ERROR: create_context failed (%d)\n", status);
        return -1;
    }

    // Create buffer for async kernel dispatch records.
    // Watermark=4096: flush frequently to avoid data loss when workers are killed.
    // SGLang/vLLM kill workers with SIGKILL — atexit handlers don't run.
    status = rocprofiler_create_buffer(
        g_ctx, 4 * 1024 * 1024, 4096,
        ROCPROFILER_BUFFER_POLICY_LOSSLESS,
        buffer_tracing_callback, nullptr, &g_buffer);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: ERROR: create_buffer failed (%d)\n", status);
        return -1;
    }

    // Create dedicated callback thread for buffer processing
    rocprofiler_callback_thread_t cb_thread;
    status = rocprofiler_create_callback_thread(&cb_thread);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: ERROR: create_callback_thread failed (%d)\n", status);
        return -1;
    }
    status = rocprofiler_assign_callback_thread(g_buffer, cb_thread);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: WARNING: assign_callback_thread failed (%d), using default\n", status);
    }

    // Register buffer tracing for kernel dispatch
    status = rocprofiler_configure_buffer_tracing_service(
        g_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
        nullptr, 0, g_buffer);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: ERROR: configure buffer tracing failed (%d)\n", status);
        return -1;
    }

    // Code object callback (synchronous — safe for symbol registration)
    rocprofiler_tracing_operation_t code_ops[] = {
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER
    };
    status = rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
        code_ops, 1, code_object_callback, nullptr);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: WARNING: code object tracing failed (%d)\n", status);
    }

    status = rocprofiler_start_context(g_ctx);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rtl: ERROR: start_context failed (%d)\n", status);
        return -1;
    }

    std::atexit(shutdown);
    fprintf(stderr, "rtl: buffer tracing initialized (pid=%d)\n", getpid());
    return 0;
}

void rtl_tool_fini(void*) {
    shutdown();
}

} // anonymous namespace

extern "C" {

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* client_id) {
    if (std::getenv("RTL_ROCP_DISABLE")) return nullptr;

    client_id->name = "rocm-trace-lite";

    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;
    fprintf(stderr, "rtl: rocprofiler-sdk v%u.%u.%u [%s] (priority=%u, pid=%d)\n",
            major, minor, patch, runtime_version ? runtime_version : "",
            priority, getpid());

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t),
        rtl_tool_init,
        rtl_tool_fini,
        nullptr
    };
    return &cfg;
}

} // extern "C"

namespace trace_db {
void rtl_trigger_shutdown() { shutdown(); }
} // namespace trace_db
