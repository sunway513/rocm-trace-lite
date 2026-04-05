#!/usr/bin/env python3
"""Minimal example: trace a GPU matmul with rocm-trace-lite.

Usage:
    # Option 1: launcher script
    ./tools/rpd_lite.sh python3 examples/trace_matmul.py

    # Option 2: direct env vars
    HSA_TOOLS_LIB=./librpd_lite.so python3 examples/trace_matmul.py

    # View results
    sqlite3 trace.rpd "SELECT * FROM top;"
    python3 tools/rpd2trace.py trace.rpd trace.json  # open in Perfetto
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

    print("Done. Check trace.rpd for results.")


if __name__ == "__main__":
    main()
