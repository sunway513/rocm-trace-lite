# ADR-001: HIP Graph Safety in rocm-trace-lite

## Status
**Accepted** — Timeout-based signal wait + batch skip + RTL_NO_INJECT escape hatch

## Problem 1: Worker signal wait crash (v0.1.x)

Binary bisect on MI300X isolated the crash to a single component:

| Test | Config | Result |
|------|--------|--------|
| Empty OnLoad | nothing | OK |
| + SQLite | open DB | OK |
| + table hooks | replace queue_create | OK |
| + intercept queue | intercept_create + callback | OK |
| + signal queue | extra barrier queue | OK |
| **+ worker signal wait** | `hsa_signal_wait_scacquire(UINT64_MAX)` | **CRASH** |

**Root cause**: Worker thread blocks indefinitely on `hsa_signal_wait_scacquire` with `UINT64_MAX` timeout. During process exit, HSA runtime destroys signals while our worker is still waiting on one. The worker accesses freed memory, causing heap corruption (`corrupted size vs. prev_size in fastbins`).

This is NOT a ROCm bug. It is our lifecycle management bug.

### Fix

Replace infinite-timeout signal wait with bounded timeout + shutdown poll:

```cpp
// Before (crashes):
hsa_signal_wait_scacquire_fn(sig, CONDITION_LT, 1, UINT64_MAX, BLOCKED);

// After (safe):
while (true) {
    val = hsa_signal_wait_scacquire_fn(sig, CONDITION_LT, 1, 100ms, BLOCKED);
    if (val < 1) break;           // kernel completed
    if (g_shutdown) { abandon; }  // exit requested
}
```

**Performance impact**: Zero in normal profiling path. Kernels complete in microseconds; the first wait always returns immediately. The 100ms timeout only triggers during shutdown, adding at most 100ms exit latency.

**Shutdown sequence**:
1. `g_shutdown = true` + `notify_all`
2. Worker exits wait loop on next timeout
3. `g_worker.join()` completes
4. Drain remaining work queue
5. Close DB

Double-shutdown prevented via `shutdown_done` atomic flag.

## Problem 2: CUDAGraph replay crashes signal injection (v0.2.x, issue #67)

Signal injection via `hsa_amd_queue_intercept_create` is incompatible with CUDAGraph/hipGraph workloads at two levels:

### Vector 1: Batch replay (count > 1)

When CUDAGraph replays, the intercept callback receives a single call with `count > 1` (e.g., 235 packets for ATOM GPT-OSS TP=8). The batch buffer contains pre-recorded AQL packets. Injecting profiling signals into these packets corrupts the graph's execution dependency chain.

Result: `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT (0x1009)`.

### Vector 2: Graph capture (stale signal handles)

During `hipStreamBeginCapture`, kernel dispatches go through the intercept callback with `count=1`. The profiler injects a profiling signal. This modified packet gets recorded into the graph. On `hipGraphLaunch` (replay), the graph submits packets containing stale profiling signal handles (already destroyed/recycled).

Result: GPU memory access fault.

### Fix: Batch skip

Skip all batch submissions (`count > 1`). No signal injection, no modification:

```cpp
const bool batch_mode = (count > 1);
if (batch_mode) {
    g_drop_not_kernel.fetch_add(count, std::memory_order_relaxed);
    writer(in_packets, count);  // pass through unmodified
    return;
}
```

Individual dispatches (`count == 1`) are still profiled normally. Graph-replayed kernels are not profiled.

### Escape hatch: RTL_NO_INJECT

`RTL_NO_INJECT=1` disables `hsa_amd_queue_intercept_create` entirely. The profiler creates a plain queue with `hsa_amd_profiling_set_profiler_enabled` instead. No kernel timestamps are collected. HIP API tracing still works.

Use when batch skip alone is insufficient (e.g., graph capture bakes stale signal handles into single-dispatch packets).

### Diagnostic logging: RTL_DEBUG

`RTL_DEBUG=1` logs per-call summary (count, device, batch skip decisions).
`RTL_DEBUG=2` adds per-packet details (AQL type, signal handle, kernel object address).

### Known limitations

- Batch skip also skips legitimate non-graph batched submissions. The HSA intercept API does not provide a way to distinguish graph replay from normal multi-packet submissions.
- Graph-captured kernels dispatched individually during capture phase still get signal injection. This is correct for normal profiling but causes stale handles on replay.
- Filed as ROCm runtime feature request: need graph-awareness in intercept callback (issue #67).
