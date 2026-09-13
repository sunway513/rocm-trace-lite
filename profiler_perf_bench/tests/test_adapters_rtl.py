"""Unit tests for adapters/rtl.py — 3 tests as per spec §6."""

import pytest
from pathlib import Path


from profiler_perf_bench.adapters.rtl import RTLAdapter


@pytest.fixture(autouse=True)
def resolved_library_path(monkeypatch):
    # These tests inspect the produced environment; they do not load a DSO.
    # Real library loading belongs to the HIP integration tests.
    monkeypatch.setattr('profiler_perf_bench.adapters.rtl._get_librtl_path',
                        lambda: '/unit-test/librtl.so')


def test_rtl_adapter_reports_missing_library(monkeypatch):
    monkeypatch.setattr('profiler_perf_bench.adapters.rtl._get_librtl_path', lambda: None)
    with pytest.raises(RuntimeError, match='librtl.so not found'):
        RTLAdapter().prepare_run([], {}, Path('/tmp'))


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
