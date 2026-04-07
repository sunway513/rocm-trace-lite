import argparse
import json
import os
import time

T_PROCESS_START = time.perf_counter()

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--hidden",
        type=int,
        default=8192,
        help="Hidden dim (tensor size = hidden * hidden)",
    )
    parser.add_argument(
        "--op",
        choices=["all_reduce", "all_gather", "both"],
        default="both",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    results = {}

    if args.op in ("all_reduce", "both"):
        tensor = torch.randn(args.hidden, args.hidden, dtype=torch.float16, device=device)
        # warmup
        for _ in range(args.warmup):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        wall_s = t1 - t0
        results["all_reduce"] = {
            "wall_s": wall_s,
            "per_iter_ms": wall_s / args.iterations * 1000,
        }

    if args.op in ("all_gather", "both"):
        # all_gather: each rank holds hidden x (hidden // world_size) chunk,
        # gather to full hidden x hidden across all ranks.
        chunk = torch.randn(
            args.hidden,
            args.hidden // world_size,
            dtype=torch.float16,
            device=device,
        )
        output_list = [torch.empty_like(chunk) for _ in range(world_size)]
        # warmup
        for _ in range(args.warmup):
            dist.all_gather(output_list, chunk)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            dist.all_gather(output_list, chunk)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        wall_s = t1 - t0
        results["all_gather"] = {
            "wall_s": wall_s,
            "per_iter_ms": wall_s / args.iterations * 1000,
        }

    # Only rank 0 prints JSON to stdout; all other ranks are silent.
    if rank == 0:
        total_wall = sum(r["wall_s"] for r in results.values())
        total_iters = sum(1 for _ in results) * args.iterations
        print(
            json.dumps(
                {
                    "workload": "nccl_comm",
                    "world_size": world_size,
                    "hidden": args.hidden,
                    "iterations": args.iterations,
                    "wall_s": total_wall,
                    "per_iter_ms": total_wall / total_iters * 1000 if total_iters else 0,
                    "total_process_s": round(time.perf_counter() - T_PROCESS_START, 6),
                    "results": results,
                }
            )
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
