"""Unit tests for adapters/rocprofv3.py and adapters/rocprof.py — 2 tests as per spec §6."""

import pytest
from pathlib import Path
from profiler_perf_bench.adapters.rocprofv3 import RocprofV3Adapter
from profiler_perf_bench.adapters.rocprof import RocprofAdapter


# Test 1: rocprofv3 adapter builds correct command prefix
def test_rocprofv3_command_prefix():
    adapter = RocprofV3Adapter()
    cmd = ["./gpu_workload", "gemm", "64", "500"]
    env = {}
    result_cmd, result_env = adapter.prepare_run(cmd, env, Path("/tmp/bench"))

    # Should be: rocprofv3 --runtime-trace -o out -- <original_cmd>
    assert result_cmd[0] == "rocprofv3"
    assert "--runtime-trace" in result_cmd
    assert "--" in result_cmd
    # Original cmd should appear after "--"
    sep_idx = result_cmd.index("--")
    assert result_cmd[sep_idx + 1:] == cmd


# Test 2: rocprof adapter builds correct command prefix
def test_rocprof_command_prefix():
    adapter = RocprofAdapter()
    cmd = ["./gpu_workload", "short", "8000"]
    env = {}
    result_cmd, result_env = adapter.prepare_run(cmd, env, Path("/tmp/bench"))

    # Should be: rocprof --hip-trace -o out.csv <original_cmd>
    assert result_cmd[0] == "rocprof"
    assert "--hip-trace" in result_cmd
    # Original cmd should appear in result_cmd
    for item in cmd:
        assert item in result_cmd
