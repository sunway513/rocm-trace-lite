"""Unit tests for sanity.py — 3 tests as per spec §6.

Covers all 4 sanity rules (rules 2+3 merged via parametrize) + dropped_reason propagation
+ run_succeeded=False exclusion from compare.
"""

import pytest
from pathlib import Path
import tempfile
import os
from profiler_perf_bench.sanity import check_sanity, SanityResult
from profiler_perf_bench.workloads.base import Level
from profiler_perf_bench.metrics import RunResult, UniversalMetrics


def _make_metrics(**kwargs):
    base: UniversalMetrics = {
        "wall_s": 1.0,
        "subprocess_s": 1.0,
        "adapter_init_s": None,
        "adapter_shutdown_s": None,
        "trace_bytes": 0,
        "peak_rss_MB": 100.0,
        "run_succeeded": True,
        "dropped_reason": None,
    }
    base.update(kwargs)
    return base


# Test 1: sanity rule 1 — exit code != 0 → dropped_reason = "crashed"
def test_sanity_rule_exit_code():
    metrics = _make_metrics()
    result = check_sanity(
        exit_code=1,
        adapter_name="none",
        workload_level=Level.L1,
        artifact_dir=Path("/tmp"),
        artifact_glob="",
        metrics=metrics,
        l3_successful_requests=None,
    )
    assert result.run_succeeded is False
    assert result.dropped_reason == "crashed"


# Test 2 (parametrized): rules 2+3 — trace file absent or too small
@pytest.mark.parametrize("scenario,expected_reason", [
    ("no_file", "no_trace_produced"),
    ("too_small", "corrupt_trace"),
])
def test_sanity_rules_trace(scenario, expected_reason, tmp_path):
    if scenario == "no_file":
        # No files in tmpdir
        artifact_dir = tmp_path
        glob_pattern = "*.trace"
    else:
        # File exists but is empty (< 100 bytes)
        trace_file = tmp_path / "output.trace"
        trace_file.write_bytes(b"tiny")
        artifact_dir = tmp_path
        glob_pattern = "*.trace"

    metrics = _make_metrics()
    result = check_sanity(
        exit_code=0,
        adapter_name="rtl",  # non-none adapter triggers trace check
        workload_level=Level.L1,
        artifact_dir=artifact_dir,
        artifact_glob=glob_pattern,
        metrics=metrics,
        l3_successful_requests=None,
    )
    assert result.run_succeeded is False
    assert result.dropped_reason == expected_reason


# Test 3: run_succeeded=False excluded from compare results
def test_failed_runs_excluded_from_compare():
    from profiler_perf_bench.report import filter_succeeded_runs

    runs = [
        RunResult("none", "L1-gemm", 0, _make_metrics(run_succeeded=True), True, None),
        RunResult("none", "L1-gemm", 1, _make_metrics(run_succeeded=False), False, "crashed"),
        RunResult("none", "L1-gemm", 2, _make_metrics(run_succeeded=True), True, None),
    ]

    succeeded = filter_succeeded_runs(runs)
    assert len(succeeded) == 2
    assert all(r.run_succeeded for r in succeeded)
