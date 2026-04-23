"""Unit tests for adapters/rtl.py — 3 tests as per spec §6."""

import pytest
import os
from pathlib import Path

try:
    import rocm_trace_lite
    _lib_path = rocm_trace_lite.get_lib_path()
    _HAS_LIBRTL = os.path.isfile(_lib_path)
except Exception:
    _HAS_LIBRTL = False

skipif_no_librtl = pytest.mark.skipif(
    not _HAS_LIBRTL,
    reason="librtl.so not found — skipping RTL adapter tests"
)


from profiler_perf_bench.adapters.rtl import RTLAdapter
from profiler_perf_bench.adapters.base import ExecutionModel


# Test 1: RTL adapter injects HSA_TOOLS_LIB and RTL_MODE into env
def test_rtl_adapter_env_injection():
    adapter = RTLAdapter(mode="lite")
    cmd = ["./gpu_workload", "gemm", "64", "500"]
    env = {}
    result_cmd, result_env = adapter.prepare_run(cmd, env, Path("/tmp"))

    assert "HSA_TOOLS_LIB" in result_env
    assert "RTL_MODE" in result_env
    assert result_env["RTL_MODE"] == "lite"
    assert result_cmd == cmd  # cmd unchanged for lite/standard mode


# Test 2: LD_PRELOAD only added for hip mode, not for lite/standard
def test_rtl_adapter_ld_preload_only_for_hip():
    lite_adapter = RTLAdapter(mode="lite")
    _, lite_env = lite_adapter.prepare_run([], {}, Path("/tmp"))
    assert "LD_PRELOAD" not in lite_env

    standard_adapter = RTLAdapter(mode="standard")
    _, std_env = standard_adapter.prepare_run([], {}, Path("/tmp"))
    assert "LD_PRELOAD" not in std_env

    hip_adapter = RTLAdapter(mode="hip")
    _, hip_env = hip_adapter.prepare_run([], {}, Path("/tmp"))
    assert "LD_PRELOAD" in hip_env


# Test 3: RTL_OUTPUT env var points into tmpdir
def test_rtl_adapter_output_in_tmpdir():
    adapter = RTLAdapter(mode="lite")
    tmpdir = Path("/tmp/bench_test_12345")
    _, env = adapter.prepare_run([], {}, tmpdir)

    # RTL_OUTPUT should reference a path under tmpdir
    rtl_output = env.get("RTL_OUTPUT", "")
    assert str(tmpdir) in rtl_output or rtl_output.startswith(str(tmpdir))
