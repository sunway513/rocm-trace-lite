/*
 * hip_api_intercept.h — Shared state for HIP API interception
 *
 * Used by hip_api_intercept.cpp (HIP wrappers) and hsa_intercept.cpp
 * (correlation ID consumption) to link HIP API calls to GPU dispatches.
 */
#pragma once

#include <cstdint>
#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace hip_api {

// Whether HIP API interception is active (set by RTL_MODE=hip)
extern std::atomic<bool> g_enabled;

// Per-stream correlation queue: maps stream pointer → FIFO of correlation IDs.
// HIP wrapper pushes, HSA completion worker pops.
class CorrelationMap {
public:
    void push(uint64_t stream, uint64_t correlation_id);
    uint64_t pop(uint64_t queue_id);  // returns 0 if empty
private:
    std::mutex mutex_;
    std::unordered_map<uint64_t, std::deque<uint64_t>> map_;
};

extern CorrelationMap g_correlation_map;

} // namespace hip_api
