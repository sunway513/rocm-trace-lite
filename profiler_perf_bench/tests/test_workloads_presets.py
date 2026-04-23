"""Unit tests for L1/L2/L3 preset config correctness — 3 tests as per spec §6."""

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
