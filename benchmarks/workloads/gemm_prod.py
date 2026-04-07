import argparse
import json
import time

T_PROCESS_START = time.perf_counter()

import torch

SHAPES = {
    "decode_moe_ffn":  {"M": 8,    "K": 7168, "N": 18432},
    "decode_attn_qkv": {"M": 8,    "K": 8192, "N": 8192},
    "decode_llama_ffn": {"M": 8,   "K": 8192, "N": 28672},
    "prefill_1k":      {"M": 1024, "K": 8192, "N": 28672},
    "prefill_4k":      {"M": 4096, "K": 8192, "N": 28672},
}


def run_shape(name, M, K, N, dtype, iterations, warmup):
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")

    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        torch.mm(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    wall_s = t1 - t0
    per_iter_ms = wall_s / iterations * 1000.0

    return {
        "M": M,
        "K": K,
        "N": N,
        "iterations": iterations,
        "wall_s": round(wall_s, 6),
        "per_iter_ms": round(per_iter_ms, 6),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Production GEMM workload for rocm-trace-lite overhead measurement"
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="all",
        choices=list(SHAPES.keys()) + ["all"],
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.shape == "all":
        results = {}
        for name, shape in SHAPES.items():
            results[name] = run_shape(
                name,
                shape["M"],
                shape["K"],
                shape["N"],
                dtype,
                args.iterations,
                args.warmup,
            )
        total_wall = sum(r["wall_s"] for r in results.values())
        total_iters = sum(r["iterations"] for r in results.values())
        print(json.dumps({
            "workload": "gemm_prod",
            "wall_s": total_wall,
            "per_iter_ms": total_wall / total_iters * 1000 if total_iters else 0,
            "results": results,
            "total_process_s": round(time.perf_counter() - T_PROCESS_START, 6),
        }))
    else:
        shape = SHAPES[args.shape]
        result = run_shape(
            args.shape,
            shape["M"],
            shape["K"],
            shape["N"],
            dtype,
            args.iterations,
            args.warmup,
        )
        print(json.dumps({
            "workload": "gemm_prod",
            "shape": args.shape,
            **result,
            "total_process_s": round(time.perf_counter() - T_PROCESS_START, 6),
        }))


if __name__ == "__main__":
    main()
