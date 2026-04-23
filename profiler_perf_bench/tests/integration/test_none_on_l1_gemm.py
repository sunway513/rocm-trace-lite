"""Integration test: none adapter on L1-gemm-small, 1 round, asserts JSON written."""

import pytest
import json
import os
from pathlib import Path


# Check if gpu_workload binary is available (needed for L1 tests)
_GPU_WORKLOAD = Path(__file__).parent.parent.parent.parent / "tests" / "gpu_workload"
_HAS_GPU_WORKLOAD = _GPU_WORKLOAD.is_file()

pytestmark = pytest.mark.skipif(
    not _HAS_GPU_WORKLOAD,
    reason=f"gpu_workload binary not found at {_GPU_WORKLOAD}"
)


def test_none_adapter_on_l1_gemm_writes_json(tmp_path):
    """Runs none adapter on L1-gemm-small, 1 round, verifies JSON output."""
    from profiler_perf_bench.adapters.none import NoneAdapter
    from profiler_perf_bench.workloads.l1.gemm_hip import GemmHipSmall
    from profiler_perf_bench.runner import BenchmarkRunner
    from profiler_perf_bench.report import format_json_report

    adapter = NoneAdapter()
    workload = GemmHipSmall(binary_path=str(_GPU_WORKLOAD))

    runner = BenchmarkRunner(adapter, workload, rounds=1)
    bench_result = runner.run()

    assert len(bench_result.rounds) == 1
    run_result = bench_result.rounds[0]
    assert run_result.run_succeeded, f"Run failed: {run_result.dropped_reason}"

    # Write JSON to tmp_path
    output_file = tmp_path / "result.json"
    report = format_json_report(bench_result.rounds, metadata={"adapter": "none", "workload": "L1-gemm-small"})
    output_file.write_text(json.dumps(report, indent=2))

    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert len(data["results"]) >= 1
    assert data["results"][0]["run_succeeded"] is True
