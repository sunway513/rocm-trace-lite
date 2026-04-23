# profiler_perf_bench

Perf-only profiler overhead benchmark suite. Compares GPU tracer/profiler overhead
across a reproducible workload set. One benchmark run = (adapter) × (workload) × (N rounds) → JSON.

## Quick Start

```bash
# List available adapters
profiler-bench adapter-list

# Run L1 sweep (RTL lite/standard/hip vs none)
profiler-bench run --config profiler_perf_bench/presets/l1_rtl_vs_none.yaml --rounds 3 --output result.json

# Regression gate — uses per-level defaults (L1: delta_ms≤50 OR pct≤15%)
profiler-bench verify --level 1

# Override with explicit threshold
profiler-bench verify --level 1 --threshold 5

# Run the steady preset (production-representative, ~2.5s run)
profiler-bench run --config profiler_perf_bench/presets/l1_rtl_vs_none.yaml --rounds 3 \
  --output result_steady.json

# Help
profiler-bench --help
```

## Adding a New Adapter

Create `profiler_perf_bench/adapters/my_adapter.py`:

```python
from profiler_perf_bench.adapters.base import ExecutionModel, ProfilerAdapter
from profiler_perf_bench.adapters.registry import global_registry
from pathlib import Path
import hashlib

@global_registry.register
class MyAdapter(ProfilerAdapter):
    name = "my_adapter"
    execution_model = ExecutionModel.EXTERNAL_WRAPPER

    def prepare_run(self, cmd, env, tmpdir):
        new_env = dict(env)
        new_env["MY_PROFILER_OUTPUT"] = str(tmpdir / "my_trace")
        return ["my-profiler-wrapper", "--"] + cmd, new_env

    def start(self, tmpdir): pass
    def stop(self): pass
    def artifact_glob(self): return "my_trace*"
    def config_hash(self): return hashlib.md5(b"my_adapter").hexdigest()
```

Then import it in your config YAML and run. No other changes needed.

## Day-1 Adapters

| Name | Backend | Execution Model |
|------|---------|----------------|
| `none` | no profiler (baseline) | — |
| `rtl` | RTL_MODE=lite | external_wrapper |
| `rtl_standard` | RTL_MODE=standard | external_wrapper |
| `rtl_hip` | RTL_MODE=hip + LD_PRELOAD | external_wrapper |
| `rocprofv3` | rocprofv3 --runtime-trace | external_wrapper |
| `rocprof` | rocprof --hip-trace | external_wrapper |
| `torch_profiler` | torch.profiler.profile | in_process_python |

## Workload Levels

| Level | Description | Deps |
|-------|-------------|------|
| L1 | HIP-native C++ binary (`gpu_workload`) | hipcc-compiled binary |
| L2 | Python torch workloads | torch + ROCm |
| L3 | ATOM/vLLM LLM serving | model weights + ATOM container |

### L1 Workload Presets

| Preset | Command | Wall time | RTL fixed-cost % |
|--------|---------|-----------|------------------|
| `L1-gemm-small` | `gemm 64 500` | ~250ms | **10-17%** (fixed-cost-dominated) |
| `L1-gemm-large` | `gemm 256 200` | ~250ms | **10-17%** (fixed-cost-dominated) |
| `L1-short-kernels` | `short 8000` | ~280ms | **9-15%** (fixed-cost-dominated) |
| `L1-multi-stream` | `multi_stream 4` | ~330ms | **8-14%** (fixed-cost-dominated) |
| `L1-gemm-steady` | `gemm 64 5000` | ~2.5s | **1-2%** (production-representative) |

**Why L1-gemm-steady exists**: The short presets show large overhead% because RTL's
fixed startup cost (~25-40ms) is divided by a ~250ms run window. This is a
**worst-case lower-bound artifact**, not representative of production workloads.

## Config YAML Format

```yaml
rounds: 3
adapters: [none, rtl, rtl_standard]
workloads: [L1-gemm-small, L1-gemm-large, L1-short-kernels, L1-multi-stream]
metadata:
  description: "My benchmark run"
```

## Running Tests

```bash
pytest profiler_perf_bench/tests -v
```

## Understanding Overhead Numbers

### Fixed startup cost vs per-kernel cost

RTL has two distinct cost components:

1. **Fixed startup cost ≈ 25-40ms per process launch** — covers:
   HSA_TOOLS_LIB dynamic load, signal pool initialization, completion worker thread
   creation, and SQLite schema init. This is a one-time cost per `gpu_workload` invocation.

2. **Per-kernel cost ≤ 1 µs/dispatch in lite mode** — the actual tracing overhead
   per GPU kernel dispatch. This is what matters for production workloads.

**RTL's per-kernel cost is ≤1 µs/dispatch in lite mode; the measured ms-level delta on
short runs reflects one-time profiler initialization, not per-kernel overhead.**

### Interpreting overhead % by run length

| Run length | Fixed cost % | Interpretation |
|-----------|-------------|----------------|
| ~250ms (L1-gemm-small) | 10-17% | `fixed_cost_dominated` — worst case artifact |
| ~2.5s (L1-gemm-steady) | 1-2% | `mixed` — startup cost becoming minor |
| 10s+ (L3 serving) | 0.24-0.4% | `workload_dominated` — production-representative |

For production representative numbers: DSR1-MXFP4 TP=4 E2E (PR#94): RTL lite = **+0.74% ITL / +1.74% TTFT**.
Laryn 1×1 transcript baseline: **"0.2-0.67%"** for production LLM serving runs.

### Per-level regression thresholds (default)

| Level | Default threshold | Rationale |
|-------|-----------------|-----------|
| L1 | delta_ms ≤ 50 **OR** pct ≤ 15% | Fixed-cost-aware; absolute budget gentler on short runs |
| L2 | pct ≤ 10% | torch workloads, moderate duration |
| L3 | pct ≤ 5% | MLPerf-representative, serving workloads |

Use `--threshold PCT` to override all levels with a single value.

### JSON output: summary fields

Each entry in `summary[]` carries:
- `delta_ms`: absolute wall-clock delta vs baseline (ms)
- `delta_pct`: percentage overhead vs baseline
- `classification`: `fixed_cost_dominated` | `mixed` | `workload_dominated`
  - `fixed_cost_dominated`: delta_ms < 50 AND baseline_ms < 1000 (startup noise)
  - `workload_dominated`: baseline_ms > 3000 (production representative)
  - `mixed`: in between

### Other notes
- `wall_s` includes process startup + HIP/HSA initialization + GPU execution + profiler shutdown.
- `trace_bytes` is reported uncompressed. RTL produces `.db` (SQLite), rocprofv3 produces
  `.rocpd` — not directly comparable without decompression.
- See `sample_results/mi355x_rtl_lite_vs_none_l1.json` for actual L1 measurements on MI355X.
