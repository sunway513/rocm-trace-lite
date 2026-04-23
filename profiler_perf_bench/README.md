# profiler_perf_bench

Perf-only profiler overhead benchmark suite. Compares GPU tracer/profiler overhead
across a reproducible workload set. One benchmark run = (adapter) × (workload) × (N rounds) → JSON.

## Quick Start

```bash
# List available adapters
profiler-bench adapter-list

# Run L1 sweep (RTL lite/standard/hip vs none)
profiler-bench run --config profiler_perf_bench/presets/l1_rtl_vs_none.yaml --rounds 3 --output result.json

# Regression gate (exits 0 if overhead < threshold%, exits 1 if regression)
profiler-bench verify --level 1 --threshold 5

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

## Notes on Overhead Measurement

- `wall_s` includes process startup + HIP/HSA initialization + GPU execution + profiler shutdown.
- RTL fixed startup cost is ~30ms per process launch. This is negligible for LLM serving
  workloads (run for minutes/hours) but appears as ~10% overhead on L1 microbenchmarks
  that run in ~0.25s. The `--threshold 5` gate is meaningful for L2/L3 serving workloads.
- `trace_bytes` is reported uncompressed. RTL produces `.db` (SQLite), rocprofv3 produces
  `.rocpd` — not directly comparable without decompression.
- See `sample_results/mi355x_rtl_lite_vs_none_l1.json` for actual L1 measurements on MI355X.
