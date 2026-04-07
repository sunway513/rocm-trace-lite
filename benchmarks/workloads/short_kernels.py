import argparse
import json
import time

import torch

T_PROCESS_START = time.perf_counter()


def main():
    parser = argparse.ArgumentParser(description="Short-kernel workload for RTL overhead measurement")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()

    t_init_start = time.perf_counter()
    t = torch.zeros(args.size, args.size, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    t_init_end = time.perf_counter()

    t_warmup_start = time.perf_counter()
    for _ in range(args.warmup):
        t.add_(1.0)
    torch.cuda.synchronize()
    t_warmup_end = time.perf_counter()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iterations):
        t.add_(1.0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    wall_s = t1 - t0
    per_iter_ms = wall_s / args.iterations * 1000.0
    t_end = time.perf_counter()

    print(json.dumps({
        "workload": "short_kernels",
        "iterations": args.iterations,
        "size": args.size,
        "wall_s": wall_s,
        "per_iter_ms": per_iter_ms,
        "cuda_init_s": t_init_end - t_init_start,
        "warmup_s": t_warmup_end - t_warmup_start,
        "total_process_s": t_end - T_PROCESS_START,
    }))


if __name__ == "__main__":
    main()
