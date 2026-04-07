import argparse
import json
import time

T_PROCESS_START = time.perf_counter()

import torch
import torch.nn as nn


class GPTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x):
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


def main():
    parser = argparse.ArgumentParser(description="Mixed-model workload for RTL overhead measurement")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    model = GPTBlock(args.hidden, args.heads).half().cuda()
    model.eval()

    x = torch.randn(args.batch_size, args.seq_len, args.hidden, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            model(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    wall_s = t1 - t0
    per_iter_ms = wall_s / args.iterations * 1000.0

    t_end = time.perf_counter()

    print(json.dumps({
        "workload": "mixed_model",
        "iterations": args.iterations,
        "wall_s": wall_s,
        "per_iter_ms": per_iter_ms,
        "total_process_s": t_end - T_PROCESS_START,
    }))


if __name__ == "__main__":
    main()
