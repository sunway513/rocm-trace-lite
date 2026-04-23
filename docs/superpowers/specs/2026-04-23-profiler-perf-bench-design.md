# Profiler-Perf-Bench — Design Spec

**Status**: Draft · **Author**: Peng Sun (P9) · **Date**: 2026-04-23
**Target repo phase**: B (lives inside `rocm-trace-lite/profiler_perf_bench/`), later extractable to standalone A.

## 1. Purpose (WHY)

A **perf-only** benchmark suite that compares the runtime overhead and throughput impact of GPU tracers/profilers across a reproducible workload set. This is the harness referenced in the 2026-04-23 Laryn 1×1 action item "benchmark and compare Peng's tracer, SDK, and third-party solutions with quantitative results" and is intended to be handed to a Claude-Code-proficient engineer from Laryn's team as the cross-team evaluation substrate.

**Explicitly NOT in scope**: functionality / correctness / feature-parity comparison (e.g. "does SDK capture all kernels?", "is API↔kernel correlation populated?"). Those belong to a separate functionality-bench effort.

## 2. Concrete problem this solves

Today each engineer re-invents profiler overhead measurement ad hoc:
- RTL has `benchmarks/overhead_bench.py` — torch-Python-only, RTL-biased discovery, no serving workload.
- ATOM nightly runs `atom_test.sh launch/benchmark` — tied to one LLM stack, limited profiler pluggability.
- rocprofiler-sdk benchmarks — internal, not cross-tool comparable.

Result: numbers from different sources aren't directly comparable, and adding a new profiler variant ("SDK with fix X", "Lauren's lightweight tracer v1") requires rebuilding harness plumbing.

This design unifies: **one benchmark run = (profiler adapter) × (workload preset) × (N rounds, interleaved) → comparable JSON row**.

## 3. Architecture

### 3.1 Module layout

```
rocm-trace-lite/
└── profiler_perf_bench/                          # new sub-package
    ├── __init__.py
    ├── adapters/
    │   ├── __init__.py
    │   ├── base.py         # ProfilerAdapter abstract + ExecutionModel enum
    │   ├── registry.py     # @register_adapter + AdapterRegistry
    │   ├── none.py         # baseline (no profiler)
    │   ├── rtl.py          # RTL via HSA_TOOLS_LIB / LD_PRELOAD
    │   ├── rocprofv3.py    # rocprofiler-sdk command prefix
    │   ├── rocprof.py      # legacy roctracer command prefix
    │   └── torch_profiler.py   # in-process Python via torch.profiler
    ├── workloads/
    │   ├── __init__.py
    │   ├── base.py         # Workload abstract + Level enum (L1/L2/L3)
    │   ├── l1/
    │   │   ├── gemm_hip.py
    │   │   ├── short_kernels_hip.py
    │   │   └── multi_stream_hip.py
    │   ├── l2/
    │   │   ├── gemm_torch.py
    │   │   └── inference_sim_torch.py
    │   └── l3/
    │       ├── dsr1_mxfp4_tp4.py
    │       ├── gpt_oss_tp1.py
    │       └── glm5_fp8_tp8.py
    ├── runner.py           # BenchmarkRunner: adapter × workload → Result
    ├── metrics.py          # Metric schema + extraction helpers
    ├── sanity.py           # 4 pre-condition guards
    ├── report.py           # JSON + Markdown output + regression-check
    ├── cli.py              # profiler-bench verify | run | adapter-list
    └── tests/              # pytest, unit + integration
```

### 3.2 Core abstractions

**ProfilerAdapter** (`adapters/base.py`):

```python
class ExecutionModel(Enum):
    EXTERNAL_WRAPPER = "external_wrapper"   # command prefix + env injection (RTL, rocprofv3, rocprof)
    IN_PROCESS_PYTHON = "in_process_python" # torch.profiler, kineto

class ProfilerAdapter(abc.ABC):
    name: str                      # "rtl", "rocprofv3", "torch_profiler", "none"
    execution_model: ExecutionModel

    # External-wrapper contract:
    def prepare_run(self, cmd: list[str], env: dict, tmpdir: Path) -> tuple[list, dict]:
        """Return (modified_cmd, modified_env). No subprocess launch."""

    # In-process-python contract:
    def start(self, tmpdir: Path) -> None: ...
    def stop(self) -> None: ...
    # Used as context manager when an in-process adapter is selected.

    # Common:
    def artifact_glob(self) -> str: ...   # glob pattern for trace outputs under tmpdir
    def config_hash(self) -> str: ...     # deterministic hash of adapter config for reporting
```

Only one of the two contracts is meaningful per adapter; `execution_model` dispatches which path `BenchmarkRunner` uses.

**Workload** (`workloads/base.py`):

```python
class Level(Enum):
    L1 = 1   # HIP-native C++ binary, <60s, no Python deps beyond hipcc-built binary
    L2 = 2   # Python (torch) workload, 1-3 min, requires torch+ROCm
    L3 = 3   # N2N LLM serving — ATOM/vLLM server + benchmark_serving client; 10-30 min, requires model weights

class Workload(abc.ABC):
    name: str
    level: Level
    requires: list[str]     # e.g., ["hipcc", "torch", "/data/models/..."], checked before run

    def cmd(self) -> list[str]: ...
    def env(self) -> dict[str, str]: ...
    def ready_probe(self) -> Optional[Callable[[], bool]]: ...    # None = no probe; L3 provides curl /health
    def client_cmd(self) -> Optional[list[str]]: ...              # None for L1/L2; L3 provides benchmark_serving cmd
    def parse_metrics(self, stdout: str, stderr: str, artifact_dir: Path) -> dict[str, float]: ...
```

L1 and L2 collapse `cmd()` + `parse_metrics()`; L3 uses all four + spawns two processes (server + client).

**BenchmarkRunner** (`runner.py`):

```python
class BenchmarkRunner:
    def __init__(self, adapter: ProfilerAdapter, workload: Workload, rounds: int = 3):
        ...
    def run_once(self) -> RunResult: ...
    def run(self) -> BenchResult:     # runs rounds × run_once, returns aggregate
        ...
```

Interleaving is done at a higher level (`compare.py`, see §3.4).

### 3.3 Metrics schema (perf-only)

`metrics.py` defines `UniversalMetrics` (TypedDict), `L3Metrics`, and `RunResult`:

```python
class UniversalMetrics(TypedDict):
    wall_s: float                  # full end-to-end wall of the run including profiler startup/shutdown
    subprocess_s: float            # subprocess.Popen wall (equal to wall_s for in_process adapters)
    adapter_init_s: float | None   # time from subprocess start to profiler ready signal; None if unobservable
    adapter_shutdown_s: float | None
    trace_bytes: int               # total bytes of profiler artifacts
    peak_rss_MB: float
    run_succeeded: bool            # sanity gate result
    dropped_reason: str | None     # set iff run_succeeded=False

class L3Metrics(TypedDict, total=False):
    ttft_ms_mean: float
    ttft_ms_median: float
    ttft_ms_p99: float
    itl_ms_mean: float
    itl_ms_median: float
    itl_ms_p99: float
    tpot_ms_mean: float
    tpot_ms_median: float
    tpot_ms_p99: float
    e2e_latency_ms_mean: float
    output_tokens_per_sec: float
    request_throughput_rps: float
    successful_requests: int
    total_requests: int
    bench_duration_s: float
```

Optional 2nd-tier fields (`avg_gpu_util_pct`, `avg_gpu_power_W`, `avg_hbm_bw_GBps`) live under `extension: dict[str, float]` in the schema but are **not collected in v1**. Callers may populate via custom workload subclasses.

### 3.4 Sanity guards (gate only, 4 fixed checks)

`sanity.py` runs after each `run_once`:

1. `subprocess exit == 0` → else `dropped_reason = "crashed"`
2. If `adapter.name != "none"`: artifact glob matches ≥1 file → else `dropped_reason = "no_trace_produced"`
3. If artifact is present: file size > 100 bytes → else `dropped_reason = "corrupt_trace"`
4. If `workload.level == L3`: `successful_requests > 0` → else `dropped_reason = "server_never_served"`

Runs with `run_succeeded=False` are recorded in JSON (for debugging) but **excluded from comparison tables and regression checks**.

### 3.5 Comparison + regression (`compare.py`, inside `report.py`)

- `compare(branches, workloads, modes, rounds)` executes a sweep with **interleaved per-round branch order** to cancel system drift.
- Output: paired median-delta table (each cell pairs branch-X and branch-Y at the same (workload, mode, round)).
- `check_regression(threshold_pct=5.0)` walks the comparison and raises `RegressionDetected` if any cell's paired-median delta on any universal/L3 metric exceeds the threshold.

### 3.6 CLI (`cli.py`)

Three subcommands:

| Command | Purpose | Primary user |
|---------|---------|--------------|
| `profiler-bench verify [--threshold 5] [--level 1,2]` | one-shot regression gate; exits 0 green, 1 fail | kernel engineer / CI |
| `profiler-bench run --config <yaml>` | sweep runner; writes `result.json` + `report.md` | cross-team engineer (Laryn's person) |
| `profiler-bench adapter-list` | dump all registered adapters + their execution model | profiler author discovering the plug point |
| (Python API) `BenchmarkRunner(adapter, workload).run()` | programmatic use | profiler developer A/B-ing their change |

## 4. Day-1 presets

### 4.1 L1 — HIP-native C++ (always runnable)

Reuses `rocm-trace-lite/tests/gpu_workload.hip` modes. The workload is the **same binary** across adapters; only the adapter's env/cmd changes.

- `L1-gemm-small`: `gemm 64 500`
- `L1-gemm-large`: `gemm 256 200`
- `L1-short-kernels`: `short 8000`
- `L1-multi-stream`: `multi_stream 4`

### 4.2 L2 — Python torch workloads (skip-if-no-torch)

Reuses `rocm-trace-lite/benchmarks/workloads/*.py`. Skip cleanly on hosts where torch can't see the GPU (gfx950 + torch 2.7.1+rocm6.2.4 is the known gap).

- `L2-gemm-torch`: `workloads/gemm.py`
- `L2-inference-sim`: `workloads/inference_sim.py`

### 4.3 L3 — ATOM/vLLM serving (opt-in, steep deps)

Configs mirror `ROCm/ATOM/.github/benchmark/models.json` nightly entries.

- `L3-dsr1-mxfp4-tp4`: `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4`, TP=4, fp8 KV, ISL=1024 OSL=1024 conc=4 (matches my 2026-04-20 PR#94 validation run)
- `L3-gpt-oss-tp1`: `openai/gpt-oss-120b`, TP=1, fp8 KV (documents the ATOM block_tables pre-existing bug; runs only if workaround env present)
- `L3-glm5-fp8-tp8`: `zai-org/GLM-5-FP8`, TP=8 (skipped on single-GPU hosts)

## 5. Day-1 adapters

| Adapter | Backend | Execution model | Required on host |
|---------|---------|-----------------|------------------|
| `none` | no profiler | — | — |
| `rtl` | `HSA_TOOLS_LIB=librtl.so RTL_MODE=lite` (and variants) | external_wrapper | librtl.so |
| `rtl_standard` | same lib, `RTL_MODE=standard` | external_wrapper | librtl.so |
| `rtl_hip` | same lib, `RTL_MODE=hip` + `LD_PRELOAD` | external_wrapper | librtl.so |
| `rocprofv3` | `rocprofv3 --runtime-trace -o out -- <cmd>` | external_wrapper | rocprofv3 binary |
| `rocprof` | `rocprof --hip-trace -o out.csv <cmd>` | external_wrapper | rocprof binary |
| `torch_profiler` | `with torch.profiler.profile(...):` | in_process_python | torch |

Adding a new adapter = subclass `ProfilerAdapter`, decorate `@register_adapter`, implement either `prepare_run` (external) or `start/stop` (in-process). Target: **~30 lines** for a new external-wrapper adapter.

## 6. Testing plan (TDD, red-green-refactor enforced)

Test enforcement: each production function gated by a failing test committed **before** its implementation. No `git commit -m "implement foo"` without a prior `git commit -m "test: foo fails"` in the history for that feature.

**Per-module test count (total 33 unit + 3 integration):**

| Module | # unit tests | What they gate |
|--------|--------------|----------------|
| adapters/base.py | 4 | ExecutionModel enum exhaustive; ABC enforcement; registry lookup; name collision |
| adapters/registry.py | 2 | decorator registration; enumerate |
| adapters/rtl.py | 3 | env injection; LD_PRELOAD only for hip mode; RTL_OUTPUT points into tmpdir |
| adapters/rocprofv3.py, rocprof.py | 2 | command prefix construction |
| adapters/torch_profiler.py | 2 | start/stop context manager semantics; trace written to tmpdir |
| adapters/none.py | 1 | no-op identity |
| workloads/base.py | 4 | Level enum; cmd/env/ready_probe contract; requires-list evaluation |
| workloads/l1/*.py, l2/*.py | 2 | preset config matches canonical sources (gpu_workload.hip modes, ATOM configs) |
| workloads/l3/*.py | 1 | L3 preset config matches `models.json` entry for dsr1_mxfp4_tp4 |
| runner.py | 4 | adapter×workload dispatch; env merge precedence; wall/subprocess time measurement; rounds interleaving |
| metrics.py | 2 | schema shape; serialization roundtrip |
| sanity.py | 3 | 4 sanity rules (2 merged into 1 test via parametrize) + dropped_reason propagation + run_succeeded=False excluded from compare |
| report.py | 3 | paired-median-delta math; regression threshold; JSON/Markdown formatting |
| cli.py | 2 | verify returns exit 1 on regression; adapter-list prints all registered |

**Integration tests** (3):
- `integration/test_none_on_l1_gemm.py` — runs `none` adapter on `L1-gemm-small`, 1 round, asserts JSON written
- `integration/test_rtl_lite_on_l1_short_kernels.py` — runs rtl adapter in lite mode, asserts overhead <5%, asserts trace.db non-empty
- `integration/test_regression_check_across_rounds.py` — synthesizes 3 paired rounds with known noise, checks regression logic

## 7. Implementation ordering (for the P8 sub-agent)

Phase-gated to let each layer compile+pass tests before the next is written:

1. **Adapter layer** (base, registry, `none`) — 6 unit tests + implementation
2. **Workload layer** (base, L1 presets) — 5 unit tests + implementation
3. **Runner layer** — 4 unit tests + implementation
4. **Metrics + sanity + report** — 8 unit tests + implementation
5. **CLI** — 2 unit tests + implementation
6. **External-wrapper adapters** (rtl, rocprofv3, rocprof) — 7 unit tests + implementation
7. **In-process adapter** (torch_profiler) — 2 unit tests + implementation
8. **L2 + L3 workload presets** — 3 unit tests + implementation
9. **Integration tests** — run on MI355X, report

## 8. Risks and design guardrails

- **Fork-safety of external-wrapper adapters under L3**: PR#94 bore this out — if an adapter crashes fork-exec'd multi-proc workers, BenchmarkRunner has to tear down cleanly without leaking KFD orphans. `runner.py` must `sudo kill -9` orphan PIDs at teardown (per existing project memory `feedback_gpu_preflight.md`) or document the pre-run check.
- **GPU availability drift during long runs**: the interleaved-round strategy addresses steady-state drift; unexpected GPU-busy from other users gets caught by sanity gate `successful_requests == 0`.
- **Trace-size as proxy for disk overhead**: profilers that compress differently (RTL's gzipped `.json.gz` vs rocprofv3's raw `.rocpd`) are not apples-to-apples on `trace_bytes`. Document: `trace_bytes` is reported raw, comparisons should decompress first if both produce compressed outputs.
- **Adapter name collisions**: registry raises on duplicate registration. CI includes an adapter-list snapshot test to catch silent renames.

## 9. Non-goals / future-looking

- **Not** a functionality parity harness.
- **Not** going to try to replicate MLPerf submission rules (we use MLPerf-shape metrics but don't submit).
- **Not** shipping a dashboard UI; the Markdown report + JSON is the output. AI-Frameworks-Dashboard can consume the JSON if desired (separate PR).
- **After maturity**: extract to a standalone repo (`ROCm/profiler-perf-bench`) per the original option A, once the adapter interface is validated by ≥2 external backends (Lauren team's new tracer is the natural 2nd).

## 10. Acceptance criteria for v0.1

1. `pytest profiler_perf_bench/tests -v` shows 33 unit + 3 integration tests all green locally on MI355X.
2. `profiler-bench verify --level 1 --threshold 5` exits 0 against RTL lite mode on L1-gemm-small, L1-short-kernels, L1-multi-stream.
3. End-to-end sweep invocation documented in `profiler_perf_bench/README.md` with one copy-pasteable command.
4. Produced JSON result for at least RTL-lite vs RTL-standard vs RTL-hip vs none on L1 suite, persisted to a committed artifact under `profiler_perf_bench/sample_results/`.
5. No changes outside `profiler_perf_bench/` directory except: (a) new `profiler-bench` console-script entry in `pyproject.toml`, (b) this design doc.

### 10.2 Per-level regression thresholds (PR#96 correction addendum)

The verify command uses per-level default thresholds when `--threshold` is not specified:

| Level | Default threshold | Gate logic | Rationale |
|-------|-----------------|-----------|-----------|
| L1 | 15% pct **OR** 50ms abs | Gentler of the two gates passes | Fixed-cost-aware: RTL startup ≈ 25-40ms inflates % on <1s microbench runs |
| L2 | 10% pct | Single pct gate | torch workloads run 1-3 min, startup cost <1% |
| L3 | 5% pct | Single pct gate (MLPerf-representative) | Serving workloads, matches PR#94 E2E validation |

**Fixed startup cost context**: RTL adds ~25-40ms one-time initialization (HSA_TOOLS_LIB load,
signal pool init, completion worker thread, SQLite schema) per process launch. This is independent
of workload length. Dividing by a 250ms run gives 10-17%; dividing by a 10s+ run gives 0.24-0.4%
— consistent with the "0.2-0.67%" Peng quoted in the Laryn 1×1 transcript.

RTL's **per-kernel cost is ≤1 µs/dispatch in lite mode**. The ms-level delta on short L1 runs
reflects initialization overhead, not per-kernel overhead.

JSON `summary[]` entries carry `delta_ms`, `delta_pct`, and `classification`:
- `"fixed_cost_dominated"`: delta_ms < 50 AND baseline_ms < 1000
- `"workload_dominated"`: baseline_ms > 3000 (production representative)
- `"mixed"`: in between

The `--threshold PCT` flag overrides all per-level defaults with a single value.
