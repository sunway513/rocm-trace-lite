/*
 * rpd_lite.h — Self-contained GPU profiling tracer
 *
 * Dependencies: libhsa-runtime64, libsqlite3
 * NO roctracer, NO rocprofiler-sdk, NO libamdhip64
 *
 * Interception mechanisms:
 *   - HSA: HSA_TOOLS_LIB OnLoad (API table replacement + queue intercept)
 *   - roctx: built-in shim (provides roctxRangePush/Pop/Mark symbols)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 */
#pragma once

#include <cstdint>
#include <string>
#include <sqlite3.h>

namespace trace_db {

// Clock source: monotonic nanoseconds
uint64_t tick();

// SQLite-backed trace database
class TraceDB {
public:
    bool open(const std::string& filename);
    void close();

    // Record a HIP API call (host-side)
    void record_hip_api(const char* name, const char* args,
                        uint64_t start_ns, uint64_t duration_ns,
                        uint64_t correlation_id, int pid, int tid);

    // Record a GPU kernel dispatch (device-side)
    // dispatch_info: optional "hwq=0x... wg=X,Y,Z grid=X,Y,Z" stored in completionSignal
    void record_kernel(const char* name, int device_id, uint64_t queue_id,
                       uint64_t start_ns, uint64_t end_ns,
                       uint64_t correlation_id,
                       const char* dispatch_info = nullptr);

    // Record a GPU memory copy (device-side)
    void record_copy(int src_device, int dst_device, size_t bytes,
                     uint64_t start_ns, uint64_t end_ns,
                     uint64_t correlation_id);

    // Record a roctx marker/range
    void record_roctx(const char* message, uint64_t start_ns, uint64_t duration_ns,
                      uint64_t correlation_id);

    // Flush buffered records
    void flush();

private:
    sqlite3* db_ = nullptr;
    sqlite3_stmt* stmt_api_ = nullptr;
    sqlite3_stmt* stmt_kernel_ = nullptr;
    sqlite3_stmt* stmt_copy_ = nullptr;
    sqlite3_stmt* stmt_roctx_ = nullptr;
    int batch_count_ = 0;
    uint64_t records_written_ = 0;
    uint64_t records_dropped_ = 0;

    void create_schema();
    void begin_transaction();
    void commit_transaction();
};

// Global trace database instance (lazy-initialized on first use)
TraceDB& get_trace_db();

// Check if trace DB is ready (avoid calling before init)
bool is_trace_ready();

// Global correlation ID counter
uint64_t next_correlation_id();

// ---- Optional callback hooks ----
// When set, interception code calls these instead of writing to TraceDB.
// This allows embedding (e.g., RPD tracer) to redirect events without
// modifying the interception logic.
//
// String lifetime: all const char* fields in event records point to
// thread-local or stack storage. Pointers are valid only for the
// duration of the callback invocation. Embedders must copy any strings
// they need to retain.
//
// Thread safety: callbacks must be registered before OnLoad fires
// (i.e., before any HIP/HSA call). After that, the pointers are
// read-only from hot paths via atomic loads.

struct ApiEventRecord {
    const char* name;       // valid only during callback
    const char* args;       // valid only during callback
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t correlation_id;
    int pid;
    int tid;
};

struct KernelEventRecord {
    const char* name;       // valid only during callback
    int device_id;
    uint64_t queue_id;
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t correlation_id;
    uint16_t wg_x, wg_y, wg_z;
    uint32_t grid_x, grid_y, grid_z;
};

using ApiEventCallback = void(*)(const ApiEventRecord& event, void* user_data);
using KernelEventCallback = void(*)(const KernelEventRecord& event, void* user_data);

void set_api_event_callback(ApiEventCallback cb, void* user_data);
void set_kernel_event_callback(KernelEventCallback cb, void* user_data);

ApiEventCallback get_api_event_callback();
KernelEventCallback get_kernel_event_callback();
void* get_api_event_callback_data();
void* get_kernel_event_callback_data();

// Trigger shutdown of the HSA intercept (joins worker, drains queue).
// Safe to call multiple times. Exposed so embedders can drain pending
// events before finalizing their own storage.
void rtl_trigger_shutdown();

} // namespace trace_db
