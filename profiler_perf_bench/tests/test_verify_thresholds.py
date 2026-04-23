"""Unit tests for per-level default thresholds in profiler-bench verify.

TDD RED phase — these tests MUST FAIL before implementation is added.

Per-level threshold spec (task WHAT §4):
  L1: delta_ms ≤ 50 OR delta_pct ≤ 15  (whichever is gentler)
  L2: delta_pct ≤ 10
  L3: delta_pct ≤ 5

CLI:
  --threshold flag overrides all per-level defaults.
  Without --threshold, use per-level defaults.
"""

import pytest


# ── Helper ──────────────────────────────────────────────────────────────────

def _make_run(adapter, workload, round_idx, wall_s, succeeded=True):
    from profiler_perf_bench.metrics import RunResult, UniversalMetrics
    metrics: UniversalMetrics = {
        "wall_s": wall_s,
        "subprocess_s": wall_s,
        "adapter_init_s": None,
        "adapter_shutdown_s": None,
        "trace_bytes": 0,
        "peak_rss_MB": 100.0,
        "run_succeeded": succeeded,
        "dropped_reason": None,
    }
    return RunResult(adapter, workload, round_idx, metrics, succeeded, None)


# ── L1 threshold: gentler of delta_ms ≤ 50 OR delta_pct ≤ 15 ────────────

def test_l1_default_threshold_passes_on_small_absolute_delta():
    """L1: 30ms delta on 250ms run = 12% but passes because delta_ms ≤ 50."""
    from profiler_perf_bench.report import check_regression_l1

    baseline = [_make_run("none", "L1-gemm-small", i, 0.250) for i in range(3)]
    adapter  = [_make_run("rtl",  "L1-gemm-small", i, 0.280) for i in range(3)]

    # 30ms delta < 50ms budget → should NOT raise (gentle absolute gate)
    check_regression_l1(baseline, adapter)  # no exception expected


def test_l1_default_threshold_passes_on_small_pct():
    """L1: 10% overhead passes the ≤15% pct gate."""
    from profiler_perf_bench.report import check_regression_l1

    baseline = [_make_run("none", "L1-gemm", i, 1.000) for i in range(3)]
    adapter  = [_make_run("rtl",  "L1-gemm", i, 1.100) for i in range(3)]

    # 10% < 15% → passes pct gate
    check_regression_l1(baseline, adapter)  # no exception


def test_l1_default_threshold_fails_when_both_exceeded():
    """L1: > 50ms AND > 15% → regression."""
    from profiler_perf_bench.report import check_regression_l1, RegressionDetected

    # 200ms delta on 300ms run = 66.7%, delta > 50ms too
    baseline = [_make_run("none", "L1-gemm", i, 0.300) for i in range(3)]
    adapter  = [_make_run("rtl",  "L1-gemm", i, 0.500) for i in range(3)]

    with pytest.raises(RegressionDetected):
        check_regression_l1(baseline, adapter)


def test_l1_default_threshold_passes_within_pct_even_if_abs_exceeded():
    """L1: delta_ms > 50 but delta_pct ≤ 15 → passes (gentler wins)."""
    from profiler_perf_bench.report import check_regression_l1

    # 60ms delta on 600ms baseline = 10% < 15% → passes
    baseline = [_make_run("none", "L1-gemm", i, 0.600) for i in range(3)]
    adapter  = [_make_run("rtl",  "L1-gemm", i, 0.660) for i in range(3)]

    check_regression_l1(baseline, adapter)  # no exception (10% < 15%)


# ── L2 threshold: delta_pct ≤ 10 ─────────────────────────────────────────

def test_l2_default_threshold_raises_above_10_pct():
    """L2: 12% overhead > 10% threshold → regression."""
    from profiler_perf_bench.report import check_regression_l2, RegressionDetected

    baseline = [_make_run("none", "L2-gemm", i, 1.000) for i in range(3)]
    adapter  = [_make_run("rtl",  "L2-gemm", i, 1.120) for i in range(3)]

    with pytest.raises(RegressionDetected):
        check_regression_l2(baseline, adapter)


def test_l2_default_threshold_passes_below_10_pct():
    """L2: 8% overhead < 10% threshold → passes."""
    from profiler_perf_bench.report import check_regression_l2

    baseline = [_make_run("none", "L2-gemm", i, 1.000) for i in range(3)]
    adapter  = [_make_run("rtl",  "L2-gemm", i, 1.080) for i in range(3)]

    check_regression_l2(baseline, adapter)  # no exception


# ── L3 threshold: delta_pct ≤ 5 ──────────────────────────────────────────

def test_l3_default_threshold_raises_above_5_pct():
    """L3: 6% overhead > 5% threshold → regression."""
    from profiler_perf_bench.report import check_regression_l3, RegressionDetected

    baseline = [_make_run("none", "L3-dsr1", i, 10.000) for i in range(3)]
    adapter  = [_make_run("rtl",  "L3-dsr1", i, 10.600) for i in range(3)]

    with pytest.raises(RegressionDetected):
        check_regression_l3(baseline, adapter)


def test_l3_default_threshold_passes_below_5_pct():
    """L3: 0.74% overhead < 5% threshold → passes (matches PR#94 E2E)."""
    from profiler_perf_bench.report import check_regression_l3

    baseline = [_make_run("none", "L3-dsr1", i, 10.000) for i in range(3)]
    adapter  = [_make_run("rtl",  "L3-dsr1", i, 10.074) for i in range(3)]

    check_regression_l3(baseline, adapter)  # no exception


# ── CLI --threshold flag overrides per-level defaults ─────────────────────

def test_cli_threshold_flag_overrides_default():
    """With explicit --threshold, per-level defaults are ignored."""
    from profiler_perf_bench.cli import build_parser, _get_level_threshold

    parser = build_parser()

    # With --threshold flag
    args_with = parser.parse_args(["verify", "--threshold", "20", "--level", "1"])
    assert _get_level_threshold(args_with, level=1) == 20.0

    # Without --threshold flag (uses default per-level for L1)
    args_without = parser.parse_args(["verify", "--level", "1"])
    assert _get_level_threshold(args_without, level=1) != 5.0  # no longer flat 5%


def test_cli_no_threshold_uses_per_level_defaults():
    """Without --threshold, verify uses L1=15%, L2=10%, L3=5% per-level defaults."""
    from profiler_perf_bench.cli import _get_level_threshold, build_parser

    parser = build_parser()
    args = parser.parse_args(["verify", "--level", "1"])

    # L1 default threshold should be 15% (pct gate, not 5%)
    t1 = _get_level_threshold(args, level=1)
    t2 = _get_level_threshold(args, level=2)
    t3 = _get_level_threshold(args, level=3)

    assert t1 == 15.0, f"L1 default threshold should be 15.0, got {t1}"
    assert t2 == 10.0, f"L2 default threshold should be 10.0, got {t2}"
    assert t3 == 5.0,  f"L3 default threshold should be 5.0, got {t3}"
