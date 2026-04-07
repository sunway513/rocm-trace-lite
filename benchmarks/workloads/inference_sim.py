import argparse
import json
import os
import time

T_PROCESS_START = time.perf_counter()

import torch
import torch.nn.functional as F


def _has_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class DecodeStep(torch.nn.Module):
    """Simulates one transformer decode step with MoE."""

    def __init__(
        self,
        hidden=7168,
        n_heads=56,
        head_dim=128,
        n_experts=8,
        topk=2,
        intermediate=18432,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_experts = n_experts
        self.topk = topk

        # RMSNorm weights
        self.norm1_weight = torch.nn.Parameter(torch.ones(hidden))
        self.norm2_weight = torch.nn.Parameter(torch.ones(hidden))

        # Attention projections
        self.qkv_proj = torch.nn.Linear(hidden, 3 * n_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(n_heads * head_dim, hidden, bias=False)

        # MoE gate
        self.gate = torch.nn.Linear(hidden, n_experts, bias=False)

        # Expert FFN (simplified: just one expert's weights, reused)
        self.w_gate_up = torch.nn.Linear(hidden, intermediate * 2, bias=False)
        self.w_down = torch.nn.Linear(intermediate, hidden, bias=False)

    def _rmsnorm(self, x, weight):
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + 1e-5)
        return (weight * x).to(orig_dtype)

    def forward(self, hidden_states):
        BS = hidden_states.shape[0]
        residual = hidden_states

        # 1. RMSNorm
        hidden_states = self._rmsnorm(hidden_states, self.norm1_weight)

        # 2. QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. Simplified attention (matmuls + softmax; no per-head reshape to keep kernel count clean)
        scale = self.head_dim ** -0.5
        attn = torch.mm(q, k.t()) * scale
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.mm(attn, v)

        # 4. Output projection
        attn_out = self.o_proj(attn_out)

        # 5. All-reduce (TP communication)
        if _has_distributed():
            torch.distributed.all_reduce(attn_out)

        hidden_states = residual + attn_out
        residual = hidden_states

        # 6. RMSNorm
        hidden_states = self._rmsnorm(hidden_states, self.norm2_weight)

        # 7. MoE gate + topk
        gate_logits = self.gate(hidden_states)
        topk_weights, topk_ids = torch.topk(gate_logits, self.topk, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        # 8. Expert FFN: gate_up -> silu-gating -> down
        gate_up = self.w_gate_up(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        expert_out = self.w_down(F.silu(gate) * up)

        # 9. All-reduce (TP communication)
        if _has_distributed():
            torch.distributed.all_reduce(expert_out)

        # 10. Residual add
        output = residual + expert_out
        return output


def main():
    parser = argparse.ArgumentParser(
        description="Inference decode simulation for rocm-trace-lite overhead measurement"
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of decode steps")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps (not measured)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (decode BS)")
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dim (DeepSeek R1 = 7168)"
    )
    parser.add_argument("--n_heads", type=int, default=56, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Attention head dimension")
    parser.add_argument("--n_experts", type=int, default=8, help="Number of MoE experts")
    parser.add_argument(
        "--intermediate", type=int, default=18432, help="Expert FFN intermediate dim"
    )
    args = parser.parse_args()

    # Setup distributed if environment is configured
    rank = 0
    if "RANK" in os.environ:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    model = (
        DecodeStep(
            hidden=args.hidden,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            n_experts=args.n_experts,
            intermediate=args.intermediate,
        )
        .half()
        .to(device)
        .eval()
    )

    x = torch.randn(args.batch_size, args.hidden, dtype=torch.float16, device=device)

    with torch.no_grad():
        # Warmup
        for _ in range(args.warmup):
            model(x)
        torch.cuda.synchronize()

        # Timed run
        t0 = time.perf_counter()
        for _ in range(args.steps):
            model(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    wall_s = t1 - t0

    if rank == 0:
        print(
            json.dumps(
                {
                    "workload": "inference_sim",
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "hidden": args.hidden,
                    "n_heads": args.n_heads,
                    "head_dim": args.head_dim,
                    "n_experts": args.n_experts,
                    "intermediate": args.intermediate,
                    "distributed": _has_distributed(),
                    "wall_s": round(wall_s, 6),
                    "per_iter_ms": round(wall_s / args.steps * 1000, 4),
                    "total_process_s": round(time.perf_counter() - T_PROCESS_START, 6),
                }
            )
        )

    if "RANK" in os.environ:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
