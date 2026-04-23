"""Unit tests for adapters/none.py — 1 test as per spec §6."""

import pytest
from pathlib import Path
from profiler_perf_bench.adapters.none import NoneAdapter


# Test 1: NoneAdapter is a no-op identity — prepare_run returns cmd+env unchanged
def test_none_adapter_identity():
    adapter = NoneAdapter()
    cmd = ["./gpu_workload", "gemm", "64", "500"]
    env = {"PATH": "/usr/bin", "HOME": "/home/test"}

    result_cmd, result_env = adapter.prepare_run(cmd, env, Path("/tmp"))

    assert result_cmd == cmd
    assert result_env == env
    assert adapter.name == "none"
    assert adapter.artifact_glob() == ""
