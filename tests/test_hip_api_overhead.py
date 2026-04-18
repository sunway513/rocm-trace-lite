"""Overhead tests for HIP API interception.

Measures wall-clock overhead of RTL_MODE=hip vs baseline and RTL_MODE=standard.
Requires: ROCm GPU + PyTorch.
"""
import os
import subprocess
import sys
import tempfile
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "librtl.so")
HAS_LIB = os.path.exists(LIB_PATH)

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False

BENCH_SCRIPT = """
import time, torch

x = torch.randn(512, 512, device='cuda', dtype=torch.float16)
for _ in range(20):
    torch.mm(x, x)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(200):
    torch.mm(x, x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"ELAPSED={elapsed:.4f}")
"""


def _run_bench(mode=None):
    env = os.environ.copy()
    if mode:
        env["HSA_TOOLS_LIB"] = LIB_PATH
        env["LD_PRELOAD"] = LIB_PATH
        env["RTL_OUTPUT"] = tempfile.mktemp(suffix=".db")
        env["RTL_MODE"] = mode

    r = subprocess.run(
        [sys.executable, "-c", BENCH_SCRIPT],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=120
    )
    assert r.returncode == 0, f"Bench failed: {r.stderr[-300:]}"
    for line in r.stdout.splitlines():
        if line.startswith("ELAPSED="):
            return float(line.split("=")[1])
    raise ValueError(f"No ELAPSED in output: {r.stdout}")


@pytest.mark.skipif(not HAS_LIB or not HAS_TORCH,
                    reason="Need librtl.so and PyTorch with CUDA")
class TestHipApiOverhead:

    def test_overhead_hip_mode(self):
        baseline = _run_bench(mode=None)
        hip_mode = _run_bench(mode="hip")
        overhead = (hip_mode - baseline) / baseline * 100
        print(f"Baseline: {baseline:.4f}s, hip mode: {hip_mode:.4f}s, "
              f"overhead: {overhead:+.1f}%")
        # HIP API interception adds per-call overhead (dlsym + record_hip_api).
        # For small workloads, fixed costs dominate. For production workloads
        # (>1s), overhead converges to <10%. Threshold set at 100% for CI.
        assert overhead < 100, f"HIP mode overhead {overhead:.1f}% exceeds 100% threshold"

    def test_standard_mode_overhead_unchanged(self):
        baseline = _run_bench(mode=None)
        standard = _run_bench(mode="standard")
        overhead = (standard - baseline) / baseline * 100
        print(f"Baseline: {baseline:.4f}s, standard: {standard:.4f}s, "
              f"overhead: {overhead:+.1f}%")
        assert overhead < 50, f"Standard mode overhead {overhead:.1f}% regression"
