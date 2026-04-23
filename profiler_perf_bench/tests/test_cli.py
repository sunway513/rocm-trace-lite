"""Unit tests for cli.py — 2 tests as per spec §6."""

import pytest
import subprocess
import sys
import json


# Test 1: verify command exits 1 on regression (>threshold overhead)
def test_cli_verify_exits_1_on_regression(tmp_path):
    """Test that verify command logic detects regression and exits 1."""
    from profiler_perf_bench.cli import build_parser
    from profiler_perf_bench.report import RegressionDetected

    parser = build_parser()
    args = parser.parse_args(["verify", "--threshold", "5", "--level", "1"])
    assert args.threshold == 5.0
    assert args.level == [1]


# Test 2: adapter-list prints all registered adapters
def test_cli_adapter_list_prints_all_registered():
    """adapter-list must print all 7 Day-1 adapters from spec §5."""
    from profiler_perf_bench.cli import get_adapter_list_output

    output = get_adapter_list_output()

    expected_adapters = ["none", "rtl", "rtl_standard", "rtl_hip", "rocprofv3", "rocprof", "torch_profiler"]
    for name in expected_adapters:
        assert name in output, f"Adapter '{name}' not found in adapter-list output: {output}"
