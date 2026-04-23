"""Integration test: synthesize 3 paired rounds with known noise, checks regression logic."""

import pytest
from profiler_perf_bench.metrics import RunResult, UniversalMetrics
from profiler_perf_bench.report import compute_paired_median_delta, check_regression, RegressionDetected


def _make_run(adapter, workload, round_idx, wall_s, succeeded=True):
    metrics: UniversalMetrics = {
        "wall_s": wall_s,
        "subprocess_s": wall_s,
        "adapter_init_s": None,
        "adapter_shutdown_s": None,
        "trace_bytes": 1024,
        "peak_rss_MB": 100.0,
        "run_succeeded": succeeded,
        "dropped_reason": None,
    }
    return RunResult(adapter, workload, round_idx, metrics, succeeded, None)


def test_regression_check_with_known_noise():
    """Synthesize 3 paired rounds: noise ±1% on base=1.0s, adapter=1.04s (4% overhead).
    With threshold=5%, should NOT raise. With threshold=3%, SHOULD raise.
    """
    import random
    random.seed(42)

    baseline_walls = [1.0 + random.uniform(-0.01, 0.01) for _ in range(3)]
    adapter_walls = [w * 1.04 + random.uniform(-0.01, 0.01) for w in baseline_walls]

    baseline_runs = [_make_run("none", "L1-gemm", i, w) for i, w in enumerate(baseline_walls)]
    adapter_runs = [_make_run("rtl", "L1-gemm", i, w) for i, w in enumerate(adapter_walls)]

    overhead = compute_paired_median_delta(baseline_runs, adapter_runs, metric="wall_s")
    # Should be approximately 4%
    assert 2.0 <= overhead <= 7.0, f"Unexpected overhead: {overhead}%"

    # threshold=5% → 4% overhead → OK
    check_regression(baseline_runs, adapter_runs, threshold_pct=5.0, metric="wall_s")

    # threshold=3% → 4% overhead → regression
    with pytest.raises(RegressionDetected):
        check_regression(baseline_runs, adapter_runs, threshold_pct=3.0, metric="wall_s")
