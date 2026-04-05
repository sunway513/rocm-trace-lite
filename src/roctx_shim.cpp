/*
 * roctx_shim.cpp — Self-contained roctx API implementation
 *
 * Provides roctxRangePushA, roctxRangePop, roctxMarkA, roctxRangeStartA, roctxRangeStop
 * so that applications using roctx markers get captured without linking libroctx64.
 *
 * When loaded via LD_PRELOAD, these symbols take priority over the real roctx library.
 */
#include "rpd_lite.h"

#include <cstdint>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdio>

using namespace rpd_lite;

// Thread-local roctx range stack (for push/pop)
struct RoctxEntry {
    std::string message;
    uint64_t start_ns;
    uint64_t correlation_id;
};

static thread_local std::vector<RoctxEntry> tls_roctx_stack;
static thread_local std::unordered_map<uint64_t, RoctxEntry> tls_roctx_ranges;
static std::atomic<uint64_t> g_range_id{1};

extern "C" {

// roctxMarkA — instant marker
void roctxMarkA(const char* message) {
    uint64_t t = tick();
    get_trace_db().record_roctx(message ? message : "", t, 0, next_correlation_id());
}

// roctxRangePushA — start a nested range
int roctxRangePushA(const char* message) {
    tls_roctx_stack.push_back({message ? message : "", tick(), next_correlation_id()});
    return (int)(tls_roctx_stack.size() - 1);
}

// roctxRangePop — end the most recent nested range
int roctxRangePop() {
    if (tls_roctx_stack.empty()) return -1;
    auto& entry = tls_roctx_stack.back();
    uint64_t now = tick();
    get_trace_db().record_roctx(entry.message.c_str(), entry.start_ns,
                                 now - entry.start_ns, entry.correlation_id);
    tls_roctx_stack.pop_back();
    return 0;
}

// roctxRangeStartA — start a non-nested range, returns range ID
uint64_t roctxRangeStartA(const char* message) {
    uint64_t id = g_range_id.fetch_add(1, std::memory_order_relaxed);
    tls_roctx_ranges[id] = {message ? message : "", tick(), next_correlation_id()};
    return id;
}

// roctxRangeStop — stop a non-nested range by ID
void roctxRangeStop(uint64_t id) {
    auto it = tls_roctx_ranges.find(id);
    if (it == tls_roctx_ranges.end()) {
        fprintf(stderr, "rpd_lite: roctxRangeStop: unknown range id %lu\n", id);
        return;
    }
    uint64_t now = tick();
    auto& entry = it->second;
    get_trace_db().record_roctx(entry.message.c_str(), entry.start_ns,
                                 now - entry.start_ns, entry.correlation_id);
    tls_roctx_ranges.erase(it);
}

// Legacy aliases
void roctxMark(const char* message) { roctxMarkA(message); }
int roctxRangePush(const char* message) { return roctxRangePushA(message); }
uint64_t roctxRangeStart(const char* message) { return roctxRangeStartA(message); }

} // extern "C"
