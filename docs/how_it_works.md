# How It Works

rocm-trace-lite captures GPU kernel execution data through HSA runtime interception, without any dependency on roctracer or rocprofiler-sdk.

## Architecture overview

```
Application (PyTorch, Triton, HIP, etc.)
    │
    ▼
HIP Runtime
    │
    ▼
HSA Runtime ◄── HSA_TOOLS_LIB=librtl.so
    │               │
    │               ├── OnLoad(): replace API table entries
    │               ├── my_hsa_queue_create(): intercept queue creation
    │               ├── queue_intercept_cb(): inject profiling signals
    │               └── completion_worker(): read timestamps, write DB
    ▼
GPU Hardware
```

## Interception mechanism

### 1. Library loading

When `HSA_TOOLS_LIB` is set, the ROCm HSA runtime calls `OnLoad()` during `hsa_init()`.
This gives us the HSA API function table, which we modify:

- `hsa_queue_create` → `my_hsa_queue_create` (intercept queue creation)
- `hsa_executable_freeze` → `my_hsa_executable_freeze` (capture kernel symbols)

### 2. Queue interception

Every `hsa_queue_create` call is redirected to create an **interceptible queue** via
`hsa_amd_queue_intercept_create`. This allows us to register a callback that sees every
AQL packet before it reaches the hardware:

```cpp
hsa_amd_queue_intercept_create(agent, size, type, ...queue);
hsa_amd_profiling_set_profiler_enabled(*queue, true);
hsa_amd_queue_intercept_register(*queue, queue_intercept_cb, &qi);
```

### 3. Signal injection profiling

For each kernel dispatch packet, the intercept callback:

1. **Acquires** a profiling signal from a reusable pool
2. **Saves** the original completion signal (if any)
3. **Replaces** `pkt->completion_signal` with the profiling signal
4. **Submits** the modified packet via `writer()`

```
Original packet:  [kernel_dispatch | signal=0x0    ]
Modified packet:  [kernel_dispatch | signal=prof_42]
```

### 4. Completion worker

A single background thread processes completed dispatches:

1. **Wait** on the profiling signal (100ms timeout for clean shutdown)
2. **Read** GPU timestamps via `hsa_amd_profiling_get_dispatch_time`
3. **Record** kernel name, device ID, timestamps to SQLite
4. **Forward** original completion signal (if non-null)
5. **Return** profiling signal to pool

### 5. Symbol resolution

Kernel names are captured by intercepting `hsa_executable_freeze`:

```cpp
hsa_executable_iterate_symbols(executable, symbol_iterate_cb, nullptr);
// Maps kernel_object handle → kernel name string
```

### 6. roctx shim

The library exports `roctxRangePushA`, `roctxRangePop`, `roctxMarkA`, `roctxRangeStartA`,
and `roctxRangeStop` symbols, allowing applications that use roctx markers to work
without linking `libroctx64`. Both nested (push/pop) and non-nested (start/stop) ranges
are supported.

## Signal pool design

Creating HSA signals is expensive. The signal pool avoids per-dispatch overhead:

- **Pre-allocate** 64 signals at startup
- **Grow** on demand up to 4096 maximum
- **Reuse** signals after completion (reset to initial value 1)
- **Destroy** excess signals when pool is full
- **Steady-state**: zero `hsa_signal_create` calls after warmup

## CUDAGraph / HIP graph handling

ROCm 10 includes the [ROCR staging-buffer fix](https://github.com/ROCm/rocm-systems/commit/559d48b1). RTL therefore processes multi-packet submissions normally, including graph replay, rather than discarding them based on `count > 1`. Original application completion signals are forwarded after profiling.

Standard and full modes collect every kernel dispatch. Lite applies its per-dispatch completion-signal filter to both eager and graph work, so its timing coverage remains partial. Unpatched older runtimes are unsupported; selecting lite does not restore the removed workaround.

### Profiling modes (RTL_MODE)

RTL supports three profiling modes to balance data completeness vs overhead:

| Mode | Mechanism | Behavior | Overhead |
|------|-----------|----------|----------|
| **lite** | HSA signal injection | Skip packets with existing `completion_signal` (NCCL, barriers). | ~0% |
| **standard** | HSA signal injection | Signal injection for all kernel dispatches, including graph replay. | ~2-4% |
| **full** | HSA signal injection | Same complete GPU capture as standard; retained for compatibility. | ~2-5% |
| **hip** | LD_PRELOAD + dlsym | HIP API interception via `dlsym(RTLD_NEXT)`. Captures CPU-side HIP call timings (21 APIs) alongside GPU kernel execution. No HSA queue interception. | <1% |

Set via `RTL_MODE=lite` env var or `rtl trace --mode lite` CLI flag.

### HIP API interception (RTL_MODE=hip)

When `RTL_MODE=hip`, RTL uses a fundamentally different mechanism: `LD_PRELOAD` function interposition via `dlsym(RTLD_NEXT)` to wrap 21 HIP runtime functions (hipModuleLaunchKernel, hipMemcpy, hipMalloc, hipFree, hipStreamSynchronize, etc.). Each wrapper records CPU-side start/end timestamps and a correlation ID, then forwards to the real HIP function. A thread-local re-entrancy guard prevents recursive recording during HIP runtime initialization. This mode populates the `rocpd_api` table with HIP API call timings.

### Known limitation

The HSA intercept API does not distinguish graph replay from normal multi-packet submissions. Batch size no longer changes profiling policy. All HSA profiling modes require the ROCR staging buffer fix (see [issue #67](https://github.com/sunway513/rocm-trace-lite/issues/67)).

### Why signal injection?

HIP runtime does not set `completion_signal` on most kernel dispatch AQL packets (ROCm 7.2+). HIP uses barrier packets with signals for synchronization instead. Without signal injection, 0 kernels would be captured. RTL injects profiling signals and forwards original signals after profiling.
