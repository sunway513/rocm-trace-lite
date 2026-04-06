#!/usr/bin/env python3
"""Minimal example: trace a GPU matmul with rocm-trace-lite.

Usage:
    rtl trace python3 examples/trace_matmul.py

    # View results
    rtl summary trace.db
    rtl convert trace.db -o trace.json   # open in Perfetto
"""
import torch


def main():
    device = "cuda"
    sizes = [512, 1024, 2048, 4096]

    print("Warming up...")
    x = torch.randn(256, 256, device=device)
    _ = x @ x
    torch.cuda.synchronize()

    print("Running matmuls...")
    for n in sizes:
        a = torch.randn(n, n, device=device, dtype=torch.float16)
        b = torch.randn(n, n, device=device, dtype=torch.float16)
        c = a @ b
        torch.cuda.synchronize()
        print(f"  {n}x{n} matmul done")

    print("Done. Check trace.db for results.")


if __name__ == "__main__":
    main()
