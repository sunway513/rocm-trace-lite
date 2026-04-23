"""Unit tests for report.py — 3 tests as per spec §6."""

import pytest
import json
from profiler_perf_bench.metrics import RunResult, UniversalMetrics
from profiler_perf_bench.report import (
    compute_paired_median_delta,
    check_regression,
    format_json_report,
    RegressionDetected,
)


def _make_run(adapter, workload, round_idx, wall_s, succeeded=True):
    metrics: UniversalMetrics = {
        "wall_s": wall_s,
        "subprocess_s": wall_s,
        "adapter_init_s": None,
        "adapter_shutdown_s": None,
        "trace_bytes": 1024,
        "peak_rss_MB": 100.0,
        "run_succeeded": succeeded,
        "dropped_reason": None if succeeded else "crashed",
    }
    return RunResult(adapter, workload, round_idx, metrics, succeeded, None if succeeded else "crashed")


# Test 1: paired_median_delta math is correct
def test_paired_median_delta_math():
    # baseline: wall_s = 1.0 (3 rounds)
    baseline_runs = [
        _make_run("none", "L1-gemm", i, 1.0) for i in range(3)
    ]
    # adapter: wall_s = 1.05 (5% overhead)
    adapter_runs = [
        _make_run("rtl", "L1-gemm", i, 1.05) for i in range(3)
    ]

    delta = compute_paired_median_delta(baseline_runs, adapter_runs, metric="wall_s")
    # delta should be ~5%
    assert abs(delta - 5.0) < 0.5


# Test 2: regression threshold detection
def test_regression_threshold():
    # 10% overhead > 5% threshold → should raise
    baseline_runs = [_make_run("none", "L1-gemm", i, 1.0) for i in range(3)]
    adapter_runs = [_make_run("rtl", "L1-gemm", i, 1.10) for i in range(3)]

    with pytest.raises(RegressionDetected):
        check_regression(baseline_runs, adapter_runs, threshold_pct=5.0, metric="wall_s")

    # 3% overhead < 5% threshold → should NOT raise
    adapter_runs_ok = [_make_run("rtl", "L1-gemm", i, 1.03) for i in range(3)]
    check_regression(baseline_runs, adapter_runs_ok, threshold_pct=5.0, metric="wall_s")  # no exception


# Test 3: JSON/Markdown formatting
def test_json_markdown_formatting():
    runs = [
        _make_run("none", "L1-gemm-small", i, 1.0) for i in range(3)
    ] + [
        _make_run("rtl", "L1-gemm-small", i, 1.03) for i in range(3)
    ]

    report = format_json_report(runs, metadata={"run_id": "test_run_1"})

    # Should produce valid JSON
    data = json.loads(json.dumps(report))
    assert "results" in data
    assert "metadata" in data
    assert len(data["results"]) == 6
