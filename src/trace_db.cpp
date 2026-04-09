/*
 * rpd_lite.cpp — SQLite trace database implementation
 */
#include "trace_db.h"

#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <atomic>
#include <mutex>
#include <unistd.h>
#include <sys/stat.h>

namespace trace_db {

static std::atomic<uint64_t> g_correlation_id{1};

// Timestamp source for host-side events (roctx markers).
// Uses CLOCK_MONOTONIC which on ROCm/Linux matches the HSA system
// timestamp domain (both derive from TSC). This is the same approach
// used by the original rtg_tracer in rocmProfileData.
// GPU kernel timestamps come from hsa_amd_profiling_get_dispatch_time
// which is in the same domain on AMD systems.
uint64_t tick() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

uint64_t next_correlation_id() {
    return g_correlation_id.fetch_add(1, std::memory_order_relaxed);
}

// ---- TraceDB ----

static TraceDB g_db;
static std::mutex g_db_mutex;
static std::once_flag g_init_flag;
static bool g_db_ready = false;

static void lazy_init_db() {
    const char* env_file = getenv("RTL_OUTPUT");
    if (!env_file) env_file = getenv("RPD_LITE_OUTPUT");  // backward compat
    std::string filename = env_file ? env_file : "trace.db";
    // Per-process trace file: replace %p with PID for multi-process safety
    auto pos = filename.find("%p");
    if (pos != std::string::npos) {
        filename.replace(pos, 2, std::to_string(getpid()));
    }
    // Create parent directories if needed
    auto slash = filename.rfind('/');
    if (slash != std::string::npos) {
        std::string dir = filename.substr(0, slash);
        for (size_t i = 1; i < dir.size(); i++) {
            if (dir[i] == '/') {
                dir[i] = '\0';
                mkdir(dir.c_str(), 0755);
                dir[i] = '/';
            }
        }
        mkdir(dir.c_str(), 0755);
    }
    if (g_db.open(filename)) {
        g_db_ready = true;
        fprintf(stderr, "rtl: lazy init, writing to %s\n", filename.c_str());
        // Note: flush/close is called by hsa_intercept::shutdown() which is
        // registered via atexit in OnLoad. No separate atexit handler needed.
    }
}

TraceDB& get_trace_db() {
    std::call_once(g_init_flag, lazy_init_db);
    return g_db;
}

bool is_trace_ready() { return g_db_ready; }

static const char* SCHEMA = R"SQL(
CREATE TABLE IF NOT EXISTS rocpd_string (id INTEGER PRIMARY KEY, string TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS rocpd_op (
    id INTEGER PRIMARY KEY,
    gpuId INTEGER,
    queueId INTEGER,
    sequenceId INTEGER,
    completionSignal TEXT,
    start INTEGER,
    end INTEGER,
    description_id INTEGER REFERENCES rocpd_string(id),
    opType_id INTEGER REFERENCES rocpd_string(id)
);
CREATE TABLE IF NOT EXISTS rocpd_api (
    id INTEGER PRIMARY KEY,
    pid INTEGER,
    tid INTEGER,
    start INTEGER,
    end INTEGER,
    apiName_id INTEGER REFERENCES rocpd_string(id),
    args_id INTEGER REFERENCES rocpd_string(id)
);
CREATE TABLE IF NOT EXISTS rocpd_api_ops (
    id INTEGER PRIMARY KEY,
    api_id INTEGER REFERENCES rocpd_api(id),
    op_id INTEGER REFERENCES rocpd_op(id)
);
CREATE TABLE IF NOT EXISTS rocpd_kernelapi (
    api_id INTEGER PRIMARY KEY REFERENCES rocpd_api(id),
    stream TEXT,
    gridX INTEGER, gridY INTEGER, gridZ INTEGER,
    workgroupX INTEGER, workgroupY INTEGER, workgroupZ INTEGER,
    groupSegmentSize INTEGER, privateSegmentSize INTEGER,
    kernelName_id INTEGER REFERENCES rocpd_string(id)
);
CREATE TABLE IF NOT EXISTS rocpd_copyapi (
    api_id INTEGER PRIMARY KEY REFERENCES rocpd_api(id),
    stream TEXT,
    size INTEGER,
    dst TEXT, src TEXT,
    kind INTEGER, sync INTEGER
);
CREATE TABLE IF NOT EXISTS rocpd_metadata (
    id INTEGER PRIMARY KEY,
    tag TEXT, value TEXT
);
CREATE TABLE IF NOT EXISTS rocpd_monitor (
    id INTEGER PRIMARY KEY,
    deviceType TEXT, deviceId INTEGER,
    monitorType TEXT,
    start INTEGER, end INTEGER,
    value TEXT
);

-- Views
CREATE VIEW IF NOT EXISTS top AS
SELECT
    s.string AS Name,
    COUNT(*) AS Calls,
    SUM(o.end - o.start) AS TotalNs,
    ROUND(AVG(o.end - o.start), 1) AS AvgNs,
    MIN(o.end - o.start) AS MinNs,
    MAX(o.end - o.start) AS MaxNs,
    ROUND(100.0 * SUM(o.end - o.start) / (SELECT SUM(end - start) FROM rocpd_op WHERE end > start AND gpuId >= 0), 2) AS Pct
FROM rocpd_op o
JOIN rocpd_string s ON o.description_id = s.id
WHERE o.end > o.start AND o.gpuId >= 0
GROUP BY s.string
ORDER BY TotalNs DESC;

CREATE VIEW IF NOT EXISTS busy AS
SELECT
    gpuId,
    ROUND(100.0 * SUM(end - start) / (MAX(end) - MIN(start)), 2) AS utilization_pct,
    COUNT(*) AS ops,
    SUM(end - start) AS busy_ns,
    MAX(end) - MIN(start) AS wall_ns
FROM rocpd_op
WHERE end > start
GROUP BY gpuId;
)SQL";

void TraceDB::create_schema() {
    char* err = nullptr;
    if (sqlite3_exec(db_, SCHEMA, nullptr, nullptr, &err) != SQLITE_OK) {
        fprintf(stderr, "rtl: schema error: %s\n", err);
        sqlite3_free(err);
    }
}

void TraceDB::begin_transaction() {
    char* err = nullptr;
    if (sqlite3_exec(db_, "BEGIN TRANSACTION", nullptr, nullptr, &err) != SQLITE_OK) {
        fprintf(stderr, "rtl: BEGIN failed: %s\n", err ? err : sqlite3_errmsg(db_));
        if (err) sqlite3_free(err);
    }
}

void TraceDB::commit_transaction() {
    char* err = nullptr;
    if (sqlite3_exec(db_, "COMMIT", nullptr, nullptr, &err) != SQLITE_OK) {
        fprintf(stderr, "rtl: COMMIT failed: %s\n", err ? err : sqlite3_errmsg(db_));
        if (err) sqlite3_free(err);
    }
    batch_count_ = 0;
}

bool TraceDB::open(const std::string& filename) {
    if (sqlite3_open(filename.c_str(), &db_) != SQLITE_OK) {
        fprintf(stderr, "rtl: cannot open %s: %s\n", filename.c_str(), sqlite3_errmsg(db_));
        return false;
    }
    // Performance pragmas
    sqlite3_exec(db_, "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA synchronous=OFF", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA cache_size=-65536", nullptr, nullptr, nullptr);

    create_schema();

    // Prepare statements — check each for errors
    auto prepare = [&](const char* sql, sqlite3_stmt** stmt, const char* label) -> bool {
        int rc = sqlite3_prepare_v2(db_, sql, -1, stmt, nullptr);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "rtl: prepare '%s' failed: %s\n", label, sqlite3_errmsg(db_));
            return false;
        }
        return true;
    };

    if (!prepare("INSERT OR IGNORE INTO rocpd_string(string) VALUES(?1)",
                 &stmt_api_, "string_insert"))
        return false;

    if (!prepare("INSERT INTO rocpd_api(pid,tid,start,end,apiName_id,args_id) "
                 "VALUES(?1,?2,?3,?4,"
                 "(SELECT id FROM rocpd_string WHERE string=?5),"
                 "(SELECT id FROM rocpd_string WHERE string=?6))",
                 &stmt_kernel_, "api_insert"))
        return false;

    if (!prepare("INSERT INTO rocpd_op(gpuId,queueId,sequenceId,start,end,description_id,opType_id) "
                 "VALUES(?1,?2,0,?3,?4,"
                 "(SELECT id FROM rocpd_string WHERE string=?5),"
                 "(SELECT id FROM rocpd_string WHERE string=?6))",
                 &stmt_copy_, "op_insert"))
        return false;

    begin_transaction();
    return true;
}

void TraceDB::close() {
    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (!db_) return;
    commit_transaction();

    // Write metadata
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d", getpid());
        sqlite3_stmt* s = nullptr;
        sqlite3_prepare_v2(db_, "INSERT INTO rocpd_metadata(tag,value) VALUES(?1,?2)", -1, &s, nullptr);
        sqlite3_bind_text(s, 1, "pid", -1, SQLITE_STATIC);
        sqlite3_bind_text(s, 2, buf, -1, SQLITE_STATIC);
        sqlite3_step(s);
        sqlite3_finalize(s);
    }

    sqlite3_finalize(stmt_api_);
    sqlite3_finalize(stmt_kernel_);
    sqlite3_finalize(stmt_copy_);
    sqlite3_finalize(stmt_roctx_);
    sqlite3_close(db_);
    db_ = nullptr;
    fprintf(stderr, "rtl: trace finalized (%" PRIu64 " records written", records_written_);
    if (records_dropped_ > 0) {
        fprintf(stderr, ", %" PRIu64 " DROPPED", records_dropped_);
    }
    fprintf(stderr, ")\n");
}

static bool ensure_string(sqlite3* db, sqlite3_stmt* stmt, const char* str) {
    if (!str || !str[0]) return true;
    sqlite3_reset(stmt);
    sqlite3_bind_text(stmt, 1, str, -1, SQLITE_TRANSIENT);
    int rc = sqlite3_step(stmt);
    return (rc == SQLITE_DONE || rc == SQLITE_ROW);
}

// Helper: step a statement and return success
static bool step_ok(sqlite3_stmt* stmt) {
    int rc = sqlite3_step(stmt);
    return (rc == SQLITE_DONE || rc == SQLITE_ROW);
}

void TraceDB::flush() {
    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (!db_) return;
    if (batch_count_ > 0) {
        commit_transaction();
        begin_transaction();
    }
}

void TraceDB::record_hip_api(const char* name, const char* args,
                              uint64_t start_ns, uint64_t duration_ns,
                              uint64_t correlation_id, int pid, int tid) {
    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (!db_) return;

    const char* safe_args = (args && args[0]) ? args : "";

    ensure_string(db_, stmt_api_, name);
    ensure_string(db_, stmt_api_, safe_args);

    sqlite3_reset(stmt_kernel_);
    sqlite3_bind_int(stmt_kernel_, 1, pid);
    sqlite3_bind_int(stmt_kernel_, 2, tid);
    sqlite3_bind_int64(stmt_kernel_, 3, start_ns);
    sqlite3_bind_int64(stmt_kernel_, 4, start_ns + duration_ns);
    sqlite3_bind_text(stmt_kernel_, 5, name, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_kernel_, 6, safe_args, -1, SQLITE_TRANSIENT);
    if (step_ok(stmt_kernel_)) { ++records_written_; } else { ++records_dropped_; }

    if (++batch_count_ >= 1000) {
        commit_transaction();
        begin_transaction();
    }
}

void TraceDB::record_kernel(const char* name, int device_id, uint64_t queue_id,
                             uint64_t start_ns, uint64_t end_ns,
                             uint64_t correlation_id) {
    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (!db_) return;

    ensure_string(db_, stmt_api_, name);
    ensure_string(db_, stmt_api_, "KernelExecution");

    sqlite3_reset(stmt_copy_);
    sqlite3_bind_int(stmt_copy_, 1, device_id);
    sqlite3_bind_int64(stmt_copy_, 2, queue_id);
    sqlite3_bind_int64(stmt_copy_, 3, start_ns);
    sqlite3_bind_int64(stmt_copy_, 4, end_ns);
    sqlite3_bind_text(stmt_copy_, 5, name, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_copy_, 6, "KernelExecution", -1, SQLITE_STATIC);
    if (step_ok(stmt_copy_)) { ++records_written_; } else { ++records_dropped_; }

    if (++batch_count_ >= 1000) {
        commit_transaction();
        begin_transaction();
    }
}

void TraceDB::record_copy(int src_device, int dst_device, size_t bytes,
                           uint64_t start_ns, uint64_t end_ns,
                           uint64_t correlation_id) {
    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (!db_) return;

    char desc[128];
    snprintf(desc, sizeof(desc), "CopyDeviceToDevice:%zu", bytes);
    ensure_string(db_, stmt_api_, desc);
    ensure_string(db_, stmt_api_, "CopyDeviceToDevice");

    sqlite3_reset(stmt_copy_);
    sqlite3_bind_int(stmt_copy_, 1, src_device);
    sqlite3_bind_int64(stmt_copy_, 2, 0);
    sqlite3_bind_int64(stmt_copy_, 3, start_ns);
    sqlite3_bind_int64(stmt_copy_, 4, end_ns);
    sqlite3_bind_text(stmt_copy_, 5, desc, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_copy_, 6, "CopyDeviceToDevice", -1, SQLITE_STATIC);
    if (step_ok(stmt_copy_)) { ++records_written_; } else { ++records_dropped_; }

    if (++batch_count_ >= 1000) {
        commit_transaction();
        begin_transaction();
    }
}

void TraceDB::record_roctx(const char* message, uint64_t start_ns, uint64_t duration_ns,
                            uint64_t correlation_id) {
    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (!db_) return;

    ensure_string(db_, stmt_api_, message);
    ensure_string(db_, stmt_api_, "UserMarker");

    sqlite3_reset(stmt_copy_);
    sqlite3_bind_int(stmt_copy_, 1, -1);  // gpuId = -1 for markers
    sqlite3_bind_int64(stmt_copy_, 2, 0);
    sqlite3_bind_int64(stmt_copy_, 3, start_ns);
    sqlite3_bind_int64(stmt_copy_, 4, start_ns + duration_ns);
    sqlite3_bind_text(stmt_copy_, 5, message, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_copy_, 6, "UserMarker", -1, SQLITE_STATIC);
    if (step_ok(stmt_copy_)) { ++records_written_; } else { ++records_dropped_; }

    if (++batch_count_ >= 1000) {
        commit_transaction();
        begin_transaction();
    }
}

} // namespace trace_db
