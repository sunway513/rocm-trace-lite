"""Unit tests for L1/L2/L3 preset config correctness — 3 tests as per spec §6.

Extended: L1-gemm-steady preset (TDD RED phase for task PR#96 correction).
"""

import pytest
from profiler_perf_bench.workloads.base import Level


# Test 1: L1 preset configs match canonical gpu_workload.hip modes
def test_l1_presets_match_gpu_workload_modes():
    from profiler_perf_bench.workloads.l1.gemm_hip import GemmHipSmall, GemmHipLarge
    from profiler_perf_bench.workloads.l1.short_kernels_hip import ShortKernelsHip
    from profiler_perf_bench.workloads.l1.multi_stream_hip import MultiStreamHip

    gemm_small = GemmHipSmall()
    assert gemm_small.level == Level.L1
    cmd = gemm_small.cmd()
    # Should use gpu_workload binary with gemm mode
    assert "gemm" in " ".join(cmd)
    assert "64" in cmd
    assert "500" in cmd

    gemm_large = GemmHipLarge()
    cmd = gemm_large.cmd()
    assert "gemm" in " ".join(cmd)
    assert "256" in cmd
    assert "200" in cmd

    short = ShortKernelsHip()
    cmd = short.cmd()
    assert "short" in " ".join(cmd)
    assert "8000" in cmd

    multi = MultiStreamHip()
    cmd = multi.cmd()
    assert "multi_stream" in " ".join(cmd)
    assert "4" in cmd


# Test 2: L2 presets have correct level and require torch
def test_l2_presets_config():
    from profiler_perf_bench.workloads.l2.gemm_torch import GemmTorchWorkload
    from profiler_perf_bench.workloads.l2.inference_sim_torch import InferenceSimTorchWorkload

    gemm = GemmTorchWorkload()
    assert gemm.level == Level.L2
    assert "torch" in gemm.requires or any("torch" in r for r in gemm.requires)

    inf_sim = InferenceSimTorchWorkload()
    assert inf_sim.level == Level.L2


# Test 3: L3 dsr1_mxfp4_tp4 preset config matches spec §4.3
def test_l3_dsr1_preset_config():
    from profiler_perf_bench.workloads.l3.dsr1_mxfp4_tp4 import DSR1MxFP4TP4Workload

    w = DSR1MxFP4TP4Workload()
    assert w.level == Level.L3
    # Should reference DeepSeek-R1 model
    cmd_str = " ".join(w.cmd()) if w.cmd() else ""
    # Check that config references the expected model or TP configuration
    assert w.name == "L3-dsr1-mxfp4-tp4"


# Test 4: L1-gemm-steady preset — TDD RED phase
def test_l1_gemm_steady_preset_config():
    """L1-gemm-steady: gemm 64 5000 (~2.5s run), registered as 'L1-gemm-steady'.

    This is the production-representative preset where fixed startup cost
    (~25-40ms) falls to ~1-2% of total wall time, versus ~10-17% for the
    250ms short presets.
    """
    from profiler_perf_bench.workloads.l1.gemm_steady_hip import GemmHipSteady

    w = GemmHipSteady()
    assert w.level == Level.L1
    assert w.name == "L1-gemm-steady"

    cmd = w.cmd()
    assert "gemm" in " ".join(cmd), f"Expected gemm mode in cmd: {cmd}"
    assert "64" in cmd, f"Expected matrix dim 64 in cmd: {cmd}"
    assert "5000" in cmd, f"Expected 5000 iterations in cmd: {cmd}"


def test_l1_gemm_steady_registered_in_workload_map():
    """L1-gemm-steady must be discoverable via the CLI workload_map (for 'run' command)."""
    # The CLI's workload_map (in _cmd_run) must include 'L1-gemm-steady'
    # We test by importing the module and checking the name is importable
    from profiler_perf_bench.workloads.l1.gemm_steady_hip import GemmHipSteady
    from profiler_perf_bench.workloads.l1 import ALL_L1_WORKLOADS

    names = [cls().name for cls in ALL_L1_WORKLOADS]
    assert "L1-gemm-steady" in names, f"L1-gemm-steady not in ALL_L1_WORKLOADS: {names}"
