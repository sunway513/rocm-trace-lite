"""Unit tests for overhead cost classifier in report.py.

TDD RED phase — these tests MUST FAIL before implementation is added.

Classifier spec (per task WHAT §3):
  classify_overhead(delta_ms, baseline_ms) -> str
  - "fixed_cost_dominated"  if delta_ms < 50 AND baseline_ms < 1000
  - "workload_dominated"    if baseline_ms > 3000
  - "mixed"                 otherwise
"""

import pytest


def test_classifier_fixed_cost_dominated():
    """Small delta + short baseline → fixed_cost_dominated."""
    from profiler_perf_bench.report import classify_overhead

    # delta_ms < 50 AND baseline_ms < 1000 → fixed_cost_dominated
    assert classify_overhead(delta_ms=30.0, baseline_ms=240.0) == "fixed_cost_dominated"
    assert classify_overhead(delta_ms=1.0, baseline_ms=999.0) == "fixed_cost_dominated"
    # boundary: delta_ms == 49 AND baseline_ms == 999 → fixed_cost_dominated
    assert classify_overhead(delta_ms=49.9, baseline_ms=999.0) == "fixed_cost_dominated"


def test_classifier_workload_dominated():
    """Long baseline (>3000ms) → workload_dominated regardless of delta."""
    from profiler_perf_bench.report import classify_overhead

    # baseline_ms > 3000 → workload_dominated
    assert classify_overhead(delta_ms=100.0, baseline_ms=10000.0) == "workload_dominated"
    assert classify_overhead(delta_ms=30.0, baseline_ms=3001.0) == "workload_dominated"
    # Even zero delta on long baseline
    assert classify_overhead(delta_ms=0.0, baseline_ms=5000.0) == "workload_dominated"


def test_classifier_mixed():
    """Cases that are neither fixed_cost_dominated nor workload_dominated → mixed."""
    from profiler_perf_bench.report import classify_overhead

    # delta_ms >= 50 AND baseline_ms <= 3000 → mixed
    assert classify_overhead(delta_ms=50.0, baseline_ms=500.0) == "mixed"
    assert classify_overhead(delta_ms=200.0, baseline_ms=1500.0) == "mixed"
    # delta_ms < 50 but baseline_ms >= 1000 AND baseline_ms <= 3000 → mixed
    assert classify_overhead(delta_ms=20.0, baseline_ms=1000.0) == "mixed"
    assert classify_overhead(delta_ms=40.0, baseline_ms=2000.0) == "mixed"


def test_classifier_boundary_conditions():
    """Exact boundary values follow the strict inequality rules."""
    from profiler_perf_bench.report import classify_overhead

    # delta_ms == 50 (NOT < 50) → mixed when baseline < 1000
    assert classify_overhead(delta_ms=50.0, baseline_ms=500.0) == "mixed"

    # baseline_ms == 3000 (NOT > 3000) → mixed when delta < 50
    assert classify_overhead(delta_ms=30.0, baseline_ms=3000.0) == "mixed"

    # baseline_ms == 3001 → workload_dominated
    assert classify_overhead(delta_ms=30.0, baseline_ms=3001.0) == "workload_dominated"


def test_format_json_report_includes_classification():
    """format_json_report summary blocks must include delta_ms, delta_pct, classification."""
    from profiler_perf_bench.metrics import RunResult, UniversalMetrics
    from profiler_perf_bench.report import format_json_report_with_deltas

    def _make_run(adapter, workload, round_idx, wall_s):
        metrics: UniversalMetrics = {
            "wall_s": wall_s,
            "subprocess_s": wall_s,
            "adapter_init_s": None,
            "adapter_shutdown_s": None,
            "trace_bytes": 0,
            "peak_rss_MB": 100.0,
            "run_succeeded": True,
            "dropped_reason": None,
        }
        return RunResult(adapter, workload, round_idx, metrics, True, None)

    # none baseline ~240ms, rtl ~270ms → delta ~30ms → fixed_cost_dominated
    runs = (
        [_make_run("none", "L1-gemm-small", i, 0.240) for i in range(3)]
        + [_make_run("rtl", "L1-gemm-small", i, 0.270) for i in range(3)]
    )

    report = format_json_report_with_deltas(runs, baseline_adapter="none")
    summary = report["summary"]

    # summary must be a list of dicts with the required fields
    assert isinstance(summary, list), f"Expected list, got {type(summary)}"
    rtl_entry = next(
        (s for s in summary if s["adapter_name"] == "rtl" and s["workload_name"] == "L1-gemm-small"),
        None,
    )
    assert rtl_entry is not None, "No rtl/L1-gemm-small entry in summary"
    assert "delta_ms" in rtl_entry, "Missing delta_ms"
    assert "delta_pct" in rtl_entry, "Missing delta_pct"
    assert "classification" in rtl_entry, "Missing classification"

    # delta ~30ms on ~240ms baseline → fixed_cost_dominated
    assert abs(rtl_entry["delta_ms"] - 30.0) < 5.0, f"delta_ms mismatch: {rtl_entry['delta_ms']}"
    assert rtl_entry["classification"] == "fixed_cost_dominated"
