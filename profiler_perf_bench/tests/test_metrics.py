"""Unit tests for metrics.py — 2 tests as per spec §6."""

import pytest
import json
from profiler_perf_bench.metrics import UniversalMetrics, RunResult, BenchResult


# Test 1: UniversalMetrics schema has required fields
def test_universal_metrics_schema_shape():
    metrics: UniversalMetrics = {
        "wall_s": 1.23,
        "subprocess_s": 1.20,
        "adapter_init_s": None,
        "adapter_shutdown_s": None,
        "trace_bytes": 0,
        "peak_rss_MB": 100.5,
        "run_succeeded": True,
        "dropped_reason": None,
    }
    # Required keys
    for key in ["wall_s", "subprocess_s", "adapter_init_s", "adapter_shutdown_s",
                "trace_bytes", "peak_rss_MB", "run_succeeded", "dropped_reason"]:
        assert key in metrics


# Test 2: RunResult serializes to/from JSON (roundtrip)
def test_run_result_serialization_roundtrip():
    metrics: UniversalMetrics = {
        "wall_s": 2.5,
        "subprocess_s": 2.4,
        "adapter_init_s": 0.1,
        "adapter_shutdown_s": 0.05,
        "trace_bytes": 4096,
        "peak_rss_MB": 200.0,
        "run_succeeded": True,
        "dropped_reason": None,
    }
    result = RunResult(
        adapter_name="none",
        workload_name="L1-gemm-small",
        round_idx=0,
        metrics=metrics,
        run_succeeded=True,
        dropped_reason=None,
    )

    # Serialize
    data = result.to_dict()
    json_str = json.dumps(data)

    # Deserialize
    restored = RunResult.from_dict(json.loads(json_str))
    assert restored.adapter_name == result.adapter_name
    assert restored.workload_name == result.workload_name
    assert restored.metrics["wall_s"] == result.metrics["wall_s"]
    assert restored.run_succeeded == result.run_succeeded
