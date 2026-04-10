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

#### Phase 2: TraceDB schema changes to persist correlation

**This phase is a prerequisite for launch latency and Perfetto correlation arrows.** The current schema in `trace_db.cpp` has no persisted API↔GPU join key:

- `rocpd_op` and `rocpd_api` tables have no `correlation_id` column
- `record_hip_api`, `record_kernel`, `record_copy` accept a `correlation_id` parameter but silently drop it — no SQL binding exists
- The `rocpd_api_ops` junction table exists in the schema but is never populated

Without persistence, CPU→GPU correlation is lost when the trace is written to SQLite, which breaks both launch latency queries and Perfetto arrows.

**Schema change (additive, backward compatible):**

```sql
ALTER TABLE rocpd_op  ADD COLUMN correlation_id INTEGER DEFAULT 0;
ALTER TABLE rocpd_api ADD COLUMN correlation_id INTEGER DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_rocpd_op_corr  ON rocpd_op(correlation_id);
CREATE INDEX IF NOT EXISTS idx_rocpd_api_corr ON rocpd_api(correlation_id);
```

Existing trace readers that don't know about the column continue to work (column default = 0). New `rtl convert` / `rtl summary` can use the column when present.

**Writer changes in `trace_db.cpp`:**

- Update the prepared INSERT statements to bind `correlation_id`
- `record_hip_api`: bind the CLR profiler's slot index
- `record_kernel` / `record_copy`: bind the same slot index from `HipClrApiRecord.gpu`
- HSA mode path: bind `trace_db::next_correlation_id()` (the existing counter) so schema is populated uniformly across modes

The junction table `rocpd_api_ops` is left unused — a direct column is simpler and avoids double-write overhead in the hot path.

#### Phase 3: HIP profiler integration and record drain

On shutdown (or at an explicit drain point — see "Memory management" below):

1. Call `hipClrProfilerDisable()`. Per the upstream implementation, this calls `DrainAllDevices()` internally to flush in-flight GPU work before clearing the enable flag.
2. Call `hipClrProfilerGetRecords(&records, &count)` to obtain the flat record array.
3. For each record, write **both** an API row and a GPU row, sharing the same `correlation_id` (the CLR slot index):
   - `trace_db.record_hip_api(api_name, "", cpu_start_ns, cpu_end_ns - cpu_start_ns, correlation_id, pid, thread_id)`
   - If `has_gpu_activity` and `op == OP_ID_DISPATCH`: `trace_db.record_kernel(kernel_name, device_id, queue_id, gpu_begin_ns, gpu_end_ns, correlation_id)`
   - If `has_gpu_activity` and `op == OP_ID_COPY`: `trace_db.record_copy(device_id, -1, bytes, gpu_begin_ns, gpu_end_ns, correlation_id)`
4. Call `hipClrProfilerReset()` to free CLR-side buffers.

#### Phase 4: CLI + env var

- `rtl trace --mode hip python3 model.py` or `RTL_MODE=hip`
- Update CLI choices: `choices=["default", "lite", "full", "hip"]`
- Update `how_it_works.md` documentation

#### Phase 5: Perfetto output enrichment

With the schema change from Phase 2, the converter can emit correlated events:

- **CPU process** (pid=1024): HIP API spans per thread (hipLaunchKernel, hipMemcpyAsync, etc.) from `rocpd_api`
- **GPU process** (pid=device_id): kernel execution spans per queue from `rocpd_op`
- **Correlation arrows**: Chrome Trace flow events (`ph:"s"` / `ph:"f"`) joining rows where `rocpd_api.correlation_id = rocpd_op.correlation_id`
- **Launch latency**: visible as the gap between CPU and GPU spans on the same flow ID

SQL for the converter's join:

```sql
SELECT a.start AS cpu_start, a.end AS cpu_end,
       o.start AS gpu_begin, o.end AS gpu_end,
       a.correlation_id AS corr_id,
       api.string AS api_name, op.string AS kernel_name
FROM rocpd_api a
JOIN rocpd_op  o  ON a.correlation_id = o.correlation_id
JOIN rocpd_string api ON a.apiName_id = api.id
JOIN rocpd_string op  ON o.description_id = op.id
WHERE a.correlation_id != 0;
```

`cmd_convert.py` falls back to the pre-change behavior (independent tables) when correlation_id is 0, so older traces still render.

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
| `src/trace_db.h` | No public API change — signature already has `correlation_id` parameter. |
| `src/trace_db.cpp` | **Required change.** Add `correlation_id INTEGER` column to `rocpd_op` and `rocpd_api` in the `SCHEMA` DDL. Update the `INSERT INTO rocpd_api` and `INSERT INTO rocpd_op` prepared statements to bind the parameter. Add indexes on the new column. Today the parameter is silently dropped — this must be fixed before correlation-based queries can work. HSA-mode writers already pass a counter-based ID, so the same path lights up both modes. |
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

Important: `hipLaunchKernel` is **asynchronous**. It returns after enqueueing the packet, typically well before the kernel finishes on the GPU. That means `cpu_end_launch < gpu_end` is the normal ordering, not an error. The validation tests below are written to reflect async semantics correctly.

```
Test 1 — Causality (async launch with bounding sync)
  For each iteration:
    record launch API:   cpu_start_L, cpu_end_L  (hipLaunchKernel)
    record sync  API:    cpu_start_S, cpu_end_S  (hipDeviceSynchronize)
    record GPU activity: gpu_begin, gpu_end      (attached to launch record)

  For every i in 1..1000:
    Assert: cpu_start_L < cpu_end_L                (API duration, trivially true)
    Assert: cpu_start_L < gpu_begin                (causality — GPU can't start
                                                    before the launch call)
    Assert: gpu_begin   < gpu_end                  (kernel execution)
    Assert: gpu_end     < cpu_end_S                (the sync call must not
                                                    return until after the
                                                    kernel completes)
    Note: NO assertion on gpu_end vs cpu_end_L —
          gpu_end > cpu_end_L is the expected async ordering.

  If cpu_start_L > gpu_begin OR gpu_end > cpu_end_S:
    clock domains are not aligned (or the CLR profiler is mis-stamping).
    All cross-domain metrics must be marked unreliable.

Test 2 — Known-duration kernel
  Launch a spin kernel with known 1ms duration (from rocprof ground truth)
  Assert: |gpu_end - gpu_begin - 1ms| / 1ms < 1%
  Confirms GPU-side timing is trustworthy.

Test 3 — Cross-domain sanity
  Async-launch 1000 noop kernels back-to-back (no per-call sync).
  Assert: for all i, gpu_begin[i] >= cpu_start_L[i]  (causality)
  Assert: median(gpu_begin - cpu_start_L) < 100μs    (reasonable magnitude)
  Assert: for all i, gpu_end[i] >= gpu_begin[i]      (self-consistency)

Test 4 — Cross-tool agreement
  Run the same 100-kernel workload under rocprof and under RTL_MODE=hip.
  Assert: median absolute difference on matched kernels' begin/end < 5μs.
```

Tests 1 and 2 are gating. Tests 3 and 4 are strongly recommended. These are added to `tests/test_gpu_hip.py` as a dedicated `test_hip_launch_latency_validation` suite.

Note on Test 1 implementation: the bounding relationship requires both the launch record and the sync record to be captured. The CLR profiler wraps every HIP API call via its dispatch table, so both records will be present in the `HipClrApiRecord` stream and can be paired by interleaving order within a single thread.

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

4. **Memory management for long-running workloads**: CLR profiler accumulates records in memory (chunked allocation, 10000 records/chunk). For a long vLLM serving run this can reach hundreds of MB. A naive "timer thread drains every N seconds" plan is **unsafe** per the upstream contract: the `hip_clr_profiler_ext.h` documentation states the returned buffer is profiler-owned, remains valid only until `Reset()` or unload, and that callers "should process records before issuing further HIP calls when profiling is active." Draining while the app is still issuing HIP calls risks invalidating the export buffer or racing with record insertion.

  Plan of record:
  - **Default (v1)**: shutdown-only drain. Simple, safe, correct. Sufficient for bounded workloads (benchmarks, test runs, CI).
  - **Soak test as the empirical check**: the 30-minute vLLM run in the validation plan measures actual memory growth. If peak RSS growth stays below budget (say < 500MB for a typical serving run), shutdown-only is good enough and we ship v1 as-is.
  - **If soak fails (v2)**: implement a cooperative "stop / drain / resume" cycle gated on a quiescence signal. Options:
    1. Explicit user trigger (roctx marker "rtl_drain" that the HSA-side roctx shim can see, triggering `Disable → GetRecords → Reset → Enable`). The app guarantees quiescence by placing the marker at a known-idle point.
    2. Piggyback on existing sync points: intercept `hipDeviceSynchronize` from the dispatch table and drain immediately after (device is known-idle at that moment).
    3. Request a streaming/callback API upstream (German's v2) — the only truly concurrent-safe option.
  - **What we are NOT doing**: a background timer thread calling `GetRecords`/`Reset` while the app runs. That contradicts the upstream contract and will race.

  Decision: ship v1 with shutdown-only, use soak test to decide if v2 is needed before GA.

5. **When will German's patch land in mainline ROCm?** This determines whether RTL can rely on the API or must treat it as experimental. Action: Peng to confirm with German.

6. **Clock-domain alignment — go/no-go for launch latency claim.** The launch latency analysis section is the selling point of HIP mode for e2e production diagnostics, but it depends on CPU and GPU timestamps being in the same domain. Must be validated with Test 1 / Test 2 from the launch latency validation protocol before the feature is advertised. If validation fails, downgrade to "HIP API CPU duration only" and file a ROCR issue.

## References

- [Issue #79](https://github.com/sunway513/rocm-trace-lite/issues/79): Feature request + v2 API proposal
- [ROCm/rocm-systems@5dc10a8](https://github.com/ROCm/rocm-systems/commit/5dc10a8): German's CLR profiler implementation
- [ADR-001](./ADR-001-hipgraph-safety.md): HIP Graph safety in RTL (context for why HIP-level profiling matters)
- Teams chat "New profiling callback proposal" (2026-04-09): German's feedback on correlation ID, CPU timeline, SDMA coverage
