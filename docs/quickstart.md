# Quick Start

## Basic usage

Profile any GPU workload with a single command:

```bash
rtl trace -o trace.db python3 my_model.py
```

This automatically:
1. Injects the profiler library via `HSA_TOOLS_LIB`
2. Captures all GPU kernel dispatches with timestamps
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
rtl trace -o trace.db torchrun --nproc_per_node=8 my_model.py
```

Each process writes to its own trace file (`trace_<PID>.db`), which are
automatically merged into the final output. GPU IDs are preserved across processes.

## Using roctx markers

Applications that use roctx markers are captured automatically:

```python
import ctypes
lib = ctypes.CDLL("librpd_lite.so")

lib.roctxRangePushA(b"forward_pass")
# ... GPU work ...
lib.roctxRangePop()

lib.roctxMarkA(b"checkpoint")
```

These appear as `UserMarker` events in the trace.

## Environment variable mode

For advanced control, set environment variables directly:

```bash
export HSA_TOOLS_LIB=/path/to/librpd_lite.so
export RPD_LITE_OUTPUT=my_trace.db
python3 my_model.py
```
