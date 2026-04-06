# ADR-001: HIP Graph Safety in rpd_lite

## Status
**Accepted** — Timeout-based signal wait

## Root Cause

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

## Fix

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
