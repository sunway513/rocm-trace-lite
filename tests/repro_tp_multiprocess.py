#!/usr/bin/env python3
"""Minimal TP>1 reproducer for rocm-trace-lite Issue #31.

Usage (requires 2+ AMD GPUs with ROCm):
    rpd-lite trace -o trace.db torchrun --nproc_per_node=2 tests/repro_tp_multiprocess.py

Each rank does:
  1. A matmul on its assigned GPU
  2. An RCCL all-reduce
  3. Another matmul

After tracing, run diagnose_trace.py on the per-process .db files
to check if all ranks recorded compute kernels.

This script is both the launcher helper AND the worker entry point.
When invoked via torchrun, RANK/LOCAL_RANK/WORLD_SIZE are set automatically.
"""
import os
import time


def worker():
    """Worker entry point — called by torchrun on each rank."""
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Diagnostics: print env inheritance proof
    hsa_tools = os.environ.get("HSA_TOOLS_LIB", "<NOT SET>")
    rpd_output = os.environ.get("RTL_OUTPUT", "<NOT SET>")
    print(f"[rank {rank}] PID={os.getpid()} LOCAL_RANK={local_rank} "
          f"WORLD_SIZE={world_size}", flush=True)
    print(f"[rank {rank}] HSA_TOOLS_LIB={hsa_tools}", flush=True)
    print(f"[rank {rank}] RTL_OUTPUT={rpd_output}", flush=True)

    # Set device
    torch.cuda.set_device(local_rank)  # maps to HIP device on ROCm
    device = torch.device(f"cuda:{local_rank}")

    # Init process group
    dist.init_process_group(
        backend="nccl",  # maps to RCCL on ROCm
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    # Workload: matmul -> all-reduce -> matmul
    N = 2048
    a = torch.randn(N, N, device=device, dtype=torch.float16)
    b = torch.randn(N, N, device=device, dtype=torch.float16)

    # First matmul
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

    # All-reduce (RCCL)
    dist.all_reduce(c, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Second matmul
    _ = torch.matmul(c, a)
    torch.cuda.synchronize()

    # Small sleep to let profiler flush
    time.sleep(0.5)

    dist.destroy_process_group()
    print(f"[rank {rank}] done, PID={os.getpid()}", flush=True)


if __name__ == "__main__":
    worker()
