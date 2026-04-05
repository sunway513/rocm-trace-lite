/*
 * rpd_lite.h — Self-contained GPU profiling tracer
 *
 * Dependencies: libamdhip64, libhsa-runtime64, libsqlite3, libfmt
 * NO roctracer, NO rocprofiler-sdk
 *
 * Interception mechanisms:
 *   - HSA: HSA_TOOLS_LIB OnLoad (API table replacement + queue intercept)
 *   - HIP: LD_PRELOAD + dlsym(RTLD_NEXT)
 *   - roctx: built-in shim (provides roctxRangePush/Pop/Mark symbols)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 */
#pragma once

#include <cstdint>
#include <string>
#include <sqlite3.h>

namespace rpd_lite {

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
    void record_kernel(const char* name, int device_id, uint64_t queue_id,
                       uint64_t start_ns, uint64_t end_ns,
                       uint64_t correlation_id);

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

} // namespace rpd_lite
