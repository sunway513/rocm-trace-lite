# Multi-GPU and Distributed Profiling

rocm-trace-lite supports profiling across multiple GPUs and distributed processes out of the box.

## How it works

When profiling multi-process workloads (e.g., `torchrun`, `torch.distributed.launch`):

1. **Per-process files**: Each process writes to its own trace file using PID substitution (`trace_%p.db`)
2. **Automatic merge**: After all processes exit, per-process files are merged into a single output
3. **GPU ID preservation**: Each process's `gpuId` is preserved in the merged trace

```
torchrun --nproc_per_node=8 model.py
    ├── Process 0 (PID 1234) → trace_1234.db (GPU 0)
    ├── Process 1 (PID 1235) → trace_1235.db (GPU 1)
    ├── ...
    └── Process 7 (PID 1241) → trace_1241.db (GPU 7)
         ↓ automatic merge
    trace.db (all 8 GPUs combined)
```

## Usage

```bash
# TP=8 inference
rtl trace -o trace.db torchrun --nproc_per_node=8 my_model.py

# Check per-GPU distribution
sqlite3 trace.db "SELECT gpuId, count(*) FROM rocpd_op GROUP BY gpuId;"
```

## Diagnostic tool

For troubleshooting multi-process profiling, use the diagnostic script:

```bash
# Inspect per-process files before merge
python3 tests/diagnose_trace.py trace_*.db
```

This reports per-file kernel counts, GPU IDs, and flags asymmetry between processes.

## Validated configurations

| Configuration | GPUs | Kernels captured | Status |
|--------------|------|-----------------|--------|
| TP=1 single process | 1 | 12/12 | Validated |
| TP=2 torchrun + RCCL | 2 | 728 (364+364) | Validated |
| TP=8 torchrun + RCCL | 8 | 20,648 | Validated |

## Diagnostic counters

Each process prints diagnostic counters at shutdown:

```text
=== rtl diagnostic (PID 336455) ===
  intercept calls:     3380
  signals injected:    2630
  drop (shutdown):     0
  drop (not kernel):   500
  drop (no qi):        0
  drop (sig pool):     0
  drop (ts fail):      0
  drop (ts invalid):   0
  recorded OK:         2630
====================================
```

Key indicators:
- **signals injected** should match **recorded OK** (no drops)
- **drop (not kernel)** counts non-kernel AQL packets (barriers, vendor-specific) that were not profiled.
- **drop (sig pool)** > 0 means signal pool exhaustion (increase `SIGNAL_POOL_MAX`)
- **drop (ts fail)** > 0 indicates GPU timestamp read failures

## CUDAGraph compatibility

On the validated ROCm 10 runtime, graph replay batches are processed in every mode. Standard/full capture all kernel dispatches; lite skips individual dispatches with application-owned completion signals. The obsolete batch-skip workaround and its counter have been removed.

For partial graph capture, select lite explicitly and measure its overhead on your workload:

```bash
rtl trace --mode lite -o trace.db torchrun --nproc_per_node=8 my_model.py
# or
RTL_MODE=lite rtl trace -o trace.db torchrun --nproc_per_node=8 my_model.py
```

For complete graph replay kernel capture on ROCm 10 (which includes [ROCR fix](https://github.com/ROCm/rocm-systems/commit/559d48b1)):

```bash
RTL_MODE=standard rtl trace -o trace.db torchrun --nproc_per_node=8 my_model.py
```

See `RTL_DEBUG=1` output for per-call diagnostics.
