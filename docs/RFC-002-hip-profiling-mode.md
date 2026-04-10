# RFC-002: HIP-level Profiling Mode (`RTL_MODE=hip`)

## Status
**Draft** — 2026-04-10

## Summary

Add a new `RTL_MODE=hip` profiling mode that uses the HIP CLR built-in profiler API (`hipClrProfiler*`) instead of HSA-level signal injection. This provides HIP API + GPU kernel timing with demangled kernel names, stream-level visibility, and copy tracking — without any AQL packet modification.

## Motivation

RTL's current architecture (HSA signal injection via `hsa_amd_queue_intercept`) has fundamental limitations documented in [issue #79](https://github.com/sunway513/rocm-trace-lite/issues/79):

| Problem | HSA mode | HIP mode (proposed) |
|---------|----------|---------------------|
| Multi-process crashes (ATOM/vLLM subprocess spawn) | Signal injection in child process → `0x1009` | Per-process `hipClrProfilerEnable()` — no signal injection |
| CUDAGraph interference | Must skip graph replay batches entirely | CLR profiler handles graph nodes natively |
| Kernel name resolution | `hsa_executable_iterate_symbols` + C++ demangle; Triton JIT → `kernel_0x...` | Native demangled names from HIP runtime |
| Stream visibility | HSA queue IDs only, no HIP stream mapping | Native `queue_id` with HIP-level semantics |
| SDMA copies | Seen as `__amd_rocclr_copyBuffer` with no metadata | `op=1 (copy)` with byte count |
| HIP API timeline | Not captured | CPU start/end timestamps for every HIP API call |

## Upstream dependency

German Andryeyev (CLR team) has implemented a built-in profiler in the HIP CLR runtime:

- **Commit**: [ROCm/rocm-systems@5dc10a8](https://github.com/ROCm/rocm-systems/commit/5dc10a8) (`clr: add internal profiler`)
- **Public header**: `hip/amd_detail/hip_clr_profiler_ext.h`
- **API surface** (4 functions):

```c
hipError_t hipClrProfilerEnable(void);
hipError_t hipClrProfilerDisable(void);
hipError_t hipClrProfilerGetRecords(const HipClrApiRecord** records, size_t* count);
hipError_t hipClrProfilerReset(void);
hipError_t hipClrProfilerWriteJson(const char* filepath);   // bonus: direct JSON export
```

- **Data structures**:

```c
typedef struct {
  uint32_t    op;           // 0=dispatch, 1=copy, 2=barrier
  uint64_t    begin_ns;     // GPU begin timestamp (ns)
  uint64_t    end_ns;       // GPU end timestamp (ns)
  int         device_id;
  uint64_t    queue_id;
  uint64_t    bytes;        // copy ops
  const char* kernel_name;  // dispatch ops (demangled)
} HipClrGpuActivity;

typedef struct {
  uint32_t          api_id;
  uint64_t          thread_id;
  uint64_t          start_ns;          // CPU timestamp
  uint64_t          end_ns;            // CPU timestamp
  int               has_gpu_activity;
  HipClrGpuActivity gpu;
} HipClrApiRecord;
```

- **Architecture**: 511 HIP dispatch table wrappers capture CPU timing + set `correlation_id` TLS. GPU timing comes back via `ACTIVITY_DOMAIN_HIP_OPS` callback, matched to CPU records by `correlation_id` (direct array index, no map).
- **Activation**: `GPU_CLR_PROFILE=1` env var or programmatic `hipClrProfilerEnable()`.
- **Output**: Chrome Trace Event JSON (same as Perfetto input).

### Availability timeline

This API is currently on a development branch in `ROCm/rocm-systems`. Expected merge timeline TBD — RTL should treat it as an **optional runtime capability** (dlopen + dlsym probe).

## Design

### Architecture

```
RTL_MODE=hip flow:

Application
    │
    ▼
HIP Runtime (libamdhip64.so)
    │  ├── hipClrProfilerEnable()     ← RTL calls at init
    │  ├── dispatch table wrappers    ← CPU timing (511 API wrappers)
    │  └── activity callback          ← GPU timing (correlation_id match)
    │
    ▼
HSA Runtime
    │  ← RTL still loads via HSA_TOOLS_LIB for lifecycle (OnLoad/OnUnload)
    │  ← NO queue intercept, NO signal injection, NO signal pool
    ▼
GPU Hardware
```

### RTL_MODE matrix (updated)

| Mode | Mechanism | Data captured | Overhead | CUDAGraph safe | Multi-proc safe |
|------|-----------|---------------|----------|----------------|-----------------|
| **default** | HSA signal injection, skip batch | GPU kernel timing | ~2-4% | Partial (skip graph replay) | No |
| **lite** | HSA signal injection, skip batch + has-signal | GPU kernel timing (partial) | ~0% | Yes | No |
| **full** | HSA signal injection, profile all | GPU kernel timing (complete) | ~2-5% | Yes (ROCm 7.13+) | No |
| **hip** (new) | HIP CLR profiler API | HIP API + GPU kernel + copy timing | ~1-3% | Yes | Yes |

### Implementation plan

#### Phase 1: dlopen integration in `hsa_intercept.cpp`

When `RTL_MODE=hip`, the `OnLoad()` entry point:

1. **Skip** queue intercept setup (no `hsa_amd_queue_intercept_create`, no signal pool init, no worker thread)
2. **dlopen** `libamdhip64.so` and resolve the 4 profiler symbols:
   ```cpp
   using EnableFn  = hipError_t(*)();
   using DisableFn = hipError_t(*)();
   using GetRecFn  = hipError_t(*)(const HipClrApiRecord**, size_t*);
   using ResetFn   = hipError_t(*)();

   void* hip_handle = dlopen("libamdhip64.so", RTLD_NOW | RTLD_NOLOAD);
   // RTLD_NOLOAD: only find already-loaded libamdhip64 (HIP app must have it)
   // If dlsym fails → API not available in this ROCm build, fall back to default mode
   ```
3. **Call** `hipClrProfilerEnable()` to activate profiling

#### Phase 2: data collection in `OnUnload()` / `shutdown()`

On shutdown:

1. Call `hipClrProfilerDisable()` (drains in-flight GPU work internally)
2. Call `hipClrProfilerGetRecords(&records, &count)`
3. Iterate records, write to TraceDB:
   - **`op=0` (dispatch)**: `trace_db.record_kernel(kernel_name, device_id, queue_id, gpu_begin_ns, gpu_end_ns, correlation_id)`
   - **`op=1` (copy)**: `trace_db.record_copy(device_id, -1, bytes, gpu_begin_ns, gpu_end_ns, correlation_id)`
   - **HIP API** (all records): `trace_db.record_hip_api(api_name, "", cpu_start_ns, cpu_end_ns - cpu_start_ns, correlation_id, pid, thread_id)`
4. Call `hipClrProfilerReset()` to free CLR-side buffers

#### Phase 3: CLI + env var

- `rtl trace --mode hip python3 model.py` or `RTL_MODE=hip`
- Update CLI choices: `choices=["default", "lite", "full", "hip"]`
- Update `how_it_works.md` documentation

#### Phase 4: Perfetto output enrichment

The HIP-level data enables richer Perfetto traces:

- **CPU process** (pid=1024): HIP API spans per thread (hipLaunchKernel, hipMemcpyAsync, etc.)
- **GPU process** (pid=device_id): kernel execution spans per queue
- **Correlation arrows**: CPU HIP call → GPU kernel execution (via `correlation_id`)
- **Launch latency**: `gpu_begin_ns - cpu_start_ns` visible as gap between CPU and GPU spans

This requires extending `cmd_convert.py` to emit HIP API records as Perfetto trace events. The existing `record_hip_api` table in TraceDB schema already supports this.

### New data available with `RTL_MODE=hip`

Compared to existing HSA-only modes, `RTL_MODE=hip` adds:

1. **Kernel launch latency** — `gpu.begin_ns - cpu_start_ns` per dispatch. Shows host-side overhead (Python, scheduling, queueing delay).
2. **HIP API timeline** — every HIP runtime call with CPU timing. Identifies host bottlenecks (hipMalloc, hipStreamSynchronize, hipDeviceSynchronize).
3. **SDMA copy tracking** — memcpy operations with byte count and GPU timing.
4. **Barrier tracking** — `op=2` barrier packets with GPU timing.
5. **Demangled kernel names** — native from HIP, no symbol table iteration needed. Triton JIT kernels get proper names.
6. **Stream-aware queue IDs** — queue_id from HIP level maps to stream semantics.

### What is NOT available (compared to HSA mode)

1. **Grid/block dimensions** — CLR profiler does not include launch config in `HipClrGpuActivity`. (Issue #79 v2 proposal requested this; may be added later.)
2. **HW queue ID** — not exposed by CLR profiler. (Same gap as HSA mode.)
3. **Per-kernel-instance correlation** — CLR uses sequential `correlation_id` (slot index), not the same ID space as RTL's `next_correlation_id()`. RTL must map between the two.

### Graceful fallback

If `RTL_MODE=hip` is set but `hipClrProfilerEnable` is not found (older ROCm, or CLR patch not merged):

```
rtl: RTL_MODE=hip requested but hipClrProfiler API not available
rtl: falling back to RTL_MODE=default
```

Log the fallback clearly. Do not crash.

## File changes

| File | Change |
|------|--------|
| `src/hsa_intercept.cpp` | Add `RtlMode::HIP` enum value. In `OnLoad()`: when mode is HIP, skip queue intercept/signal pool/worker init; call `hip_profiler_init()`. In `shutdown()`: call `hip_profiler_drain()`. |
| `src/hip_intercept.cpp` | Replace placeholder with HIP CLR profiler integration: `hip_profiler_init()`, `hip_profiler_drain()`, dlopen/dlsym logic, record conversion to TraceDB. |
| `src/trace_db.h` | No changes — existing `record_hip_api`, `record_kernel`, `record_copy` methods cover all needed record types. |
| `rocm_trace_lite/cli.py` | Add `"hip"` to `--mode` choices. |
| `rocm_trace_lite/cmd_trace.py` | No changes — already forwards `args.mode` to `RTL_MODE` env var. |
| `rocm_trace_lite/cmd_convert.py` | Extend Perfetto JSON output to include HIP API spans (from `rocpd_api` table) and correlation arrows. |
| `docs/how_it_works.md` | Document HIP mode architecture, data flow, and comparison table. |
| `Makefile` | No changes — `hip_intercept.cpp` already in SRCS. HIP header inclusion guarded by `#ifdef` / dlopen. |

## Alternatives considered

### 1. HIP callback API (Issue #79 original proposal)

Push-model callback (`hipRegisterKernelCompletionCallback`). More flexible for real-time streaming, but:
- Does not exist yet — would require new HIP API development
- German's pull-model API is already implemented and tested
- Can revisit as v2 if callback API materializes

### 2. Direct `GPU_CLR_PROFILE=1` env var (no RTL integration)

Users could set `GPU_CLR_PROFILE=1` themselves and get a `hip_clr_trace.json`. But:
- No SQLite trace DB (RTL's native format)
- No multi-process merge
- No `rtl summary` / `rtl convert` pipeline
- No roctx markers integration
- No unified mode system (`RTL_MODE`)

### 3. roctracer-based HIP tracing

Would give HIP API + activity records, but contradicts RTL's core design principle (zero roctracer dependency). Also being deprecated (EoS 2026 Q2).

## Kernel launch latency analysis

The HIP CLR profiler record carries 4 timestamps per kernel:

```
cpu_start_ns  — app enters hipLaunchKernel()
cpu_end_ns    — hipLaunchKernel() returns to app
gpu_begin_ns  — kernel begins execution on GPU
gpu_end_ns    — kernel finishes on GPU
```

This is a new capability — HSA mode has no CPU-side timestamps at all. It unlocks several latency decompositions, but comes with a clock-domain caveat that must be validated before we can claim launch latency as a shipped feature.

### Metrics we can compute

| Metric | Formula | Physical meaning | Confidence |
|--------|---------|------------------|------------|
| HIP API cost | `cpu_end - cpu_start` | CPU time inside HIP runtime (packet build + enqueue) | **High** — single clock, self-consistent |
| Launch-to-execution latency | `gpu_begin - cpu_start` | End-to-end launch latency — "the number everyone wants" | **Medium** — depends on cross-domain clock alignment |
| Queue-to-execution delay | `gpu_begin - cpu_end` | Time after HIP returns before GPU picks up work (queue scheduling + CP dispatch + wave launch) | **Medium** — same caveat |
| Back-to-back GPU gap | `gpu_begin[N+1] - gpu_end[N]` on same stream | GPU-perceived idle time between kernels (host-starved diagnostic) | **High** — pure GPU clock |
| Kernel duration | `gpu_end - gpu_begin` | Same as HSA mode | **High** — pure GPU clock |

### Clock-domain caveat

`cpu_*_ns` comes from `std::chrono::high_resolution_clock` (Linux: `CLOCK_MONOTONIC`). `gpu_*_ns` comes from `hsa_amd_profiling_get_dispatch_time`, translated to the system clock domain by ROCR's `TranslateTime()`/`SyncClocks()` path.

On ROCm 7.13+ Linux / MI300+, these should be in the same domain with sub-μs drift. But history says we must verify:
- Windows/WSL had a 100× scaling bug that German just fixed (part of the same commit as the CLR profiler).
- Pre-7.0 ROCm had `ticksToTime_` precision issues.
- `SyncClocks()` recalibrates on timer, so drift between calibration points can be non-zero.

Expected residual error on validated Linux/MI300+: sub-μs to a few μs. Good enough for diagnosing 10-50μs launch latencies, not good enough for ns-level attribution.

### Validation protocol (gating test — must pass before feature ships)

```
Test 1 — Causality (sync launch)
  for i in 1..1000:
    hipLaunchKernel(noop)
    hipDeviceSynchronize()
  Assert: cpu_start[i] < gpu_begin[i] < gpu_end[i] < cpu_end[i]
  If this fails: clock domains are not aligned,
                 all cross-domain metrics must be marked unreliable.

Test 2 — Known-duration kernel
  Launch a spin kernel with known 1ms duration (from rocprof ground truth)
  Assert: |gpu_end - gpu_begin - 1ms| / 1ms < 1%
  Confirms GPU-side timing is trustworthy.

Test 3 — Cross-domain sanity
  Async-launch 1000 noop kernels back-to-back (no sync).
  Assert: for all i, gpu_begin[i] > cpu_start[i]  (causality)
  Assert: median(gpu_begin - cpu_start) < 100μs  (reasonable magnitude)

Test 4 — Cross-tool agreement
  Run the same 100-kernel workload under rocprof and under RTL_MODE=hip.
  Assert: median absolute difference on matched kernels' begin/end < 5μs.
```

Tests 1 and 2 are gating. Test 3 and 4 are strongly recommended. These are added to `tests/test_gpu_hip.py` as a dedicated `test_hip_launch_latency_validation` suite.

### Go / no-go decision point

If Test 1 passes → ship launch latency as a first-class feature, document it in `how_it_works.md`, add a "Launch latency" column to `rtl summary`.

If Test 1 fails → downgrade the feature to "HIP API CPU duration only", remove cross-domain claims from the RFC, file a ROCm issue against ROCR clock translation, and continue shipping HIP mode for the other benefits (multi-process safety, stream visibility, demangled names, SDMA tracking).

### High-value use cases (assuming validation passes)

- **vLLM step decomposition**: for each decode step, how much time is Python / torch / HIP API cost vs real GPU compute? Answers "why is my GPU only 60% busy".
- **ATOM scheduler tuning**: after the scheduler hands a batch to the runtime, how long until the GPU actually starts?
- **torch.compile vs eager**: measure the actual launch latency reduction, not just end-to-end throughput.
- **Host-bound diagnosis**: `gpu_begin[N+1] - gpu_end[N] > 0` with CPU busy → host can't feed GPU fast enough. Actionable signal.
- **hipGraph win quantification**: measure the actual launch latency delta between graph replay and eager. RTL has never been able to answer this independently before — it always needed torch.profiler.

The hipGraph case is the most compelling. Current HSA mode can't capture CPU-side timestamps at all, so the question "how much does hipGraph actually save me" has been unanswerable with RTL. HIP mode fixes this.

### What we still cannot extract

| Metric | Why not |
|--------|---------|
| CP internal dispatch latency | Needs SQTT or wave counters |
| HW queue arbitration delay | Not exposed at HIP or HSA level |
| Per-wave launch timing | Needs rocprofiler-sdk |
| RCCL wait-on-barrier timing | Not a kernel — barrier packet at HSA level, invisible to HIP profiler |

For these, users still need rocprofiler-sdk / rocprof compute viewer. RTL's positioning stays the same: lightweight always-on tracer, not a deep-dive profiler.

## Overhead & validation plan

Two separate things to measure: per-feature overhead (HIP mode vs other modes) and end-to-end production workload overhead.

### Overhead benchmark matrix (`benchmarks/overhead_bench.py`)

Extend the existing benchmark to include `hip` mode. Compare all 4 columns side by side.

| Baseline | `RTL_MODE=default` | `RTL_MODE=lite` | `RTL_MODE=hip` |
|----------|--------------------|-----------------|----------------|
| no profiler | HSA inject | HSA inject, skip signaled | HIP CLR profiler |

Metrics captured per row:
- Wall-clock slowdown vs baseline (%)
- Records captured / records dropped
- Trace file size (bytes)
- Peak RSS delta vs baseline
- Per-kernel overhead (μs)

Workloads:
- `workloads/gemm.py` — long-running kernels, launch latency amortized
- `workloads/short_kernels.py` — launch-latency dominated, stress test
- `workloads/mixed_model.py` — representative mix
- `workloads/inference_sim.py` — serving-style short-kernel bursts
- `workloads/nccl_comm.py` — multi-GPU communication (check if CLR profiler sees RCCL dispatches)

### End-to-end validation

| Workload | Platform | Metrics | Pass criteria |
|----------|----------|---------|---------------|
| vLLM GPT-OSS 120B TP=8 | MI355X | throughput (tokens/s), TTFT, TPOT, peak RSS | `hip` vs no-profiler ≤ 5% throughput degradation; TTFT delta ≤ 10ms |
| ATOM DeepSeek-R1 TP=8 | MI355X | throughput, TTFT, TPOT, **no mp.spawn crash** | ≤ 5% throughput; zero `0x1009` crashes (this is the #1 blocker from issue #79) |
| PyTorch ResNet50 train, batch=256 | MI300X | steps/s, memory | ≤ 3% |
| Micro: vectoradd × 10000 | MI300X | wall clock | ≤ 2% |
| Stress: 100k short kernels | MI300X | records captured | 100% capture, 0 drops |
| Long-run soak: vLLM 30min | MI355X | peak RSS drift, records | RSS drift < 500MB (tests periodic drain) |

The long-run soak test is new and specifically targets the CLR profiler's in-memory accumulation risk (10k records/chunk × N chunks). It's the main memory-management difference between HIP mode and HSA mode, which is streaming-write to SQLite.

### Regression safety

- All existing `RTL_MODE=default/lite/full` tests must continue to pass unchanged.
- `test_source_guard.py` verifies no new link-time dependencies (dlopen is runtime, not compile-time).
- `make test-cpu` green on CI.
- Existing overhead targets for `default` / `lite` / `full` modes unchanged.

## Open questions

1. **Correlation ID mapping**: CLR profiler uses sequential slot indices as `correlation_id`. RTL uses `trace_db::next_correlation_id()`. Should we use CLR's IDs directly, or remap? Recommendation: use CLR's IDs directly — they're unique within a process, and RTL's correlation IDs are only used for DB queries.

2. **API name table**: CLR profiler provides `api_id` (uint32) and a name table `kHipClrApiNames[]`. We need to dlsym the name table, or reconstruct from the header. Recommendation: dlsym `kHipClrApiNames` and `kHipClrApiNamesCount`.

3. **Coexistence with HSA mode**: Should `RTL_MODE=hip` completely disable HSA queue intercept, or run both in parallel? Recommendation: completely disable HSA intercept in HIP mode. Running both would double-count kernels and add unnecessary overhead.

4. **Periodic drain vs shutdown-only**: CLR profiler accumulates all records in memory (chunked allocation, 10000 records/chunk). For long-running workloads this could use significant memory. Should RTL periodically call `GetRecords` + `Reset`? Recommendation: drain every N seconds (configurable, default 30s) for long-running workloads. Add a timer thread or piggyback on roctx flush intervals. The long-run soak test in the validation plan is the empirical check for whether this is necessary.

5. **When will German's patch land in mainline ROCm?** This determines whether RTL can rely on the API or must treat it as experimental. Action: Peng to confirm with German.

6. **Clock-domain alignment — go/no-go for launch latency claim.** The launch latency analysis section is the selling point of HIP mode for e2e production diagnostics, but it depends on CPU and GPU timestamps being in the same domain. Must be validated with Test 1 / Test 2 from the launch latency validation protocol before the feature is advertised. If validation fails, downgrade to "HIP API CPU duration only" and file a ROCR issue.

## References

- [Issue #79](https://github.com/sunway513/rocm-trace-lite/issues/79): Feature request + v2 API proposal
- [ROCm/rocm-systems@5dc10a8](https://github.com/ROCm/rocm-systems/commit/5dc10a8): German's CLR profiler implementation
- [ADR-001](./ADR-001-hipgraph-safety.md): HIP Graph safety in RTL (context for why HIP-level profiling matters)
- Teams chat "New profiling callback proposal" (2026-04-09): German's feedback on correlation ID, CPU timeline, SDMA coverage
