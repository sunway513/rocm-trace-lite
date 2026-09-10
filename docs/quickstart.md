# Quick Start

Install the candidate as described in [installation](installation.md).

## First trace without PyTorch

From a source checkout in the ROCm development image, compile the small HIP workload, then run it with your installed wheel:

```bash
hipcc -O2 tests/gpu_workload.hip -o /tmp/rtl-example -lpthread
cd /tmp
rtl trace --mode standard -o first-trace.db -- ./rtl-example short 100
rtl info first-trace.db
rtl summary first-trace.db
sqlite3 first-trace.db "PRAGMA integrity_check;"
```

The example submits 100 `vec_add` kernels. Check that all 100 appear before interpreting timing. Open the generated `first-trace.json.gz` in [Perfetto](https://ui.perfetto.dev). This small workload checks installation and capture; its timing is not a model performance estimate.

## Basic usage

Profile any GPU workload with a single command:

```bash
rtl trace --mode standard -o trace.db python3 my_model.py
```

This automatically:
1. Injects the profiler library via `HSA_TOOLS_LIB`
2. Captures intercepted GPU kernel dispatches with timestamps in standard mode
3. Merges per-process traces (for multi-GPU / distributed workloads)
4. Generates a summary, Perfetto JSON, and SQLite database

## View results

### Terminal summary

```bash
rtl summary trace.db
```

```text
Trace: trace.db
  GPU ops:   728

Kernel                                              Calls  Total(us)  Avg(us)      %
========================================================================================
Cijk_Ailk_Bljk_HHS_BH_MT128x128x128                  240    28252.9    117.7   21.8
ncclDevKernel_Generic                                  160    29747.8    185.9   23.0
__amd_rocclr_fillBufferAligned.kd                     7900    27929.8      3.5   21.6

GPU Utilization:
  GPU 0: 0.13% (2630 ops, 17.2ms busy)
  GPU 1: 0.11% (2430 ops, 15.0ms busy)
```

### Perfetto timeline

The `trace` command auto-generates a compressed `.json.gz` file.
Open it in [ui.perfetto.dev](https://ui.perfetto.dev) for interactive timeline visualization.

### SQL queries

The trace file is a standard SQLite database. Query it directly:

```bash
# Top 10 kernels by GPU time
sqlite3 trace.db "SELECT * FROM top LIMIT 10;"

# GPU utilization
sqlite3 trace.db "SELECT * FROM busy;"

# All GEMM kernels
sqlite3 trace.db "
  SELECT s.string, count(*), sum(o.end - o.start)/1000 as total_us
  FROM rocpd_op o
  JOIN rocpd_string s ON o.description_id = s.id
  WHERE s.string LIKE '%Cijk%'
  GROUP BY s.string
  ORDER BY total_us DESC;
"
```

## Multi-GPU / Distributed

rocm-trace-lite automatically handles multi-process workloads (e.g., `torchrun`):

```bash
rtl trace --mode standard -o trace.db torchrun --nproc_per_node=8 my_model.py
```

Each process writes to its own trace file (`trace_<PID>.db`), which are
automatically merged into the final output. GPU IDs are preserved across processes.

## Using roctx markers

Applications that use roctx markers are captured automatically:

```python
import ctypes
from rocm_trace_lite import get_lib_path
lib = ctypes.CDLL(get_lib_path())

# Nested ranges (push/pop)
lib.roctxRangePushA(b"forward_pass")
# ... GPU work ...
lib.roctxRangePop()

# Non-nested ranges (start/stop)
lib.roctxRangeStartA.restype = ctypes.c_uint64
lib.roctxRangeStop.argtypes = [ctypes.c_uint64]
rid = lib.roctxRangeStartA(b"data_loading")
# ... work ...
lib.roctxRangeStop(rid)

# Instant markers
lib.roctxMarkA(b"checkpoint")
```

These are stored as `UserMarker` records in SQLite. The current Perfetto converter does not export all ROCTX records; verify annotations in SQLite when completeness matters.

## CUDAGraph / HIP graph compatibility

On the validated ROCm 10 runtime, **standard** and **full** capture graph replay kernels. The runtime includes the staging-buffer fix needed for safe interception, so the former batch-skip workaround is removed.

**lite**, the default, still omits individual packets with an existing completion signal and can produce an incomplete graph timeline. Choose standard for completeness checks. Unpatched older runtimes are unsupported in this candidate.

## Environment variables

| Variable | Values | Description |
|----------|--------|-------------|
| `RTL_OUTPUT` | path | Output trace file (supports `%p` for PID). Alternative to `-o` flag. `RPD_LITE_OUTPUT` also accepted for backward compatibility. |
| `RTL_MODE` | `lite`, `standard`, `full`, `hip` | Profiling mode (see below) |
| `RTL_DEBUG` | `1`, `2` | Packet-level diagnostic logging (1=summary, 2=per-packet) |

### Profiling modes

| Mode | GPU timing | HIP API | Graph replay | Intended scope |
|------|-----------|---------|-------------|----------------|
| **lite** (default) | Partial | No | Partial | Sampling dispatches without an existing completion signal |
| **standard** | All intercepted kernels | No | Profiled | GPU timeline and kernel analysis |
| **full** | Same as standard | No | Profiled | Compatibility name for standard GPU coverage |
| **hip** | GPU timing plus wrapped APIs | Yes | Profiled | Selected HIP API correlation; separate validation required |

Overhead depends on launch rate, capture coverage, mode and runtime. Use `--mode standard` when validating completeness. See [measured performance and limitations](performance.md); no fixed overhead percentage applies to every workload.

## TraceLens analysis

Convert RTL traces to rocprofv3 format for TraceLens performance reports:

```bash
rtl convert trace.db --format rocprofv3 -o trace_results.json
TraceLens_generate_perf_report_rocprof --profile_json_path trace_results.json
```

This produces an Excel workbook with GPU timeline breakdown, kernel summary by category, and per-dispatch details. See [TraceLens](https://github.com/AMD-AGI/TraceLens) for installation.

## Environment variable mode

For advanced control, set environment variables directly:

```bash
export HSA_TOOLS_LIB=/path/to/librtl.so
export RTL_OUTPUT=my_trace.db
export RTL_MODE=lite    # partial dispatch coverage
python3 my_model.py
```
