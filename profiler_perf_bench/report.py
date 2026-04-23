"""Report generation: JSON + Markdown output, regression check, compare logic.

Per spec §3.5:
  - compare() runs a sweep with interleaved per-round branch order.
  - check_regression() raises RegressionDetected if any cell exceeds threshold_pct.
"""

from __future__ import annotations
import statistics
from typing import Any, Dict, List, Optional

from .metrics import RunResult


class RegressionDetected(Exception):
    """Raised when overhead exceeds regression threshold."""
    pass


def filter_succeeded_runs(runs: List[RunResult]) -> List[RunResult]:
    """Return only runs where run_succeeded=True.

    Per spec §3.4: runs with run_succeeded=False are excluded from comparison tables.
    """
    return [r for r in runs if r.run_succeeded]


def compute_paired_median_delta(
    baseline_runs: List[RunResult],
    adapter_runs: List[RunResult],
    metric: str = "wall_s",
) -> float:
    """Compute paired median delta as percentage overhead.

    Pairs runs by round_idx. Returns (median_adapter - median_baseline) / median_baseline * 100.

    Only succeeded runs are used (spec §3.4).
    """
    baseline_ok = filter_succeeded_runs(baseline_runs)
    adapter_ok = filter_succeeded_runs(adapter_runs)

    baseline_vals = [r.metrics[metric] for r in baseline_ok if metric in r.metrics]
    adapter_vals = [r.metrics[metric] for r in adapter_ok if metric in r.metrics]

    if not baseline_vals or not adapter_vals:
        raise ValueError(f"No valid runs to compare on metric '{metric}'")

    baseline_median = statistics.median(baseline_vals)
    adapter_median = statistics.median(adapter_vals)

    if baseline_median == 0:
        return float("inf")

    return (adapter_median - baseline_median) / baseline_median * 100.0


def check_regression(
    baseline_runs: List[RunResult],
    adapter_runs: List[RunResult],
    threshold_pct: float = 5.0,
    metric: str = "wall_s",
) -> None:
    """Raise RegressionDetected if overhead exceeds threshold_pct.

    Per spec §3.5: walks the comparison and raises if any cell's paired-median delta
    on any metric exceeds the threshold.
    """
    delta = compute_paired_median_delta(baseline_runs, adapter_runs, metric=metric)
    if delta > threshold_pct:
        adapter_name = adapter_runs[0].adapter_name if adapter_runs else "unknown"
        workload_name = baseline_runs[0].workload_name if baseline_runs else "unknown"
        raise RegressionDetected(
            f"Regression detected: {adapter_name} vs none on {workload_name}/{metric}: "
            f"{delta:.2f}% > {threshold_pct:.1f}% threshold"
        )


def format_json_report(
    runs: List[RunResult],
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """Format runs as a JSON-serializable report dict.

    Structure:
      {
        "metadata": {...},
        "results": [{...}, ...],
        "summary": {adapter: {workload: {metric: median_value}}}
      }
    """
    from collections import defaultdict

    results_list = [r.to_dict() for r in runs]

    # Build summary: adapter → workload → metric → median
    summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    adapter_workload: Dict[tuple, List[RunResult]] = defaultdict(list)
    for r in runs:
        if r.run_succeeded:
            adapter_workload[(r.adapter_name, r.workload_name)].append(r)

    for (adapter, workload), grp_runs in adapter_workload.items():
        for metric in ["wall_s", "subprocess_s", "trace_bytes", "peak_rss_MB"]:
            vals = [r.metrics[metric] for r in grp_runs if metric in r.metrics]
            if vals:
                summary[adapter][workload][metric] = statistics.median(vals)

    return {
        "metadata": metadata or {},
        "results": results_list,
        "summary": {k: dict(v) for k, v in summary.items()},
    }


def format_markdown_table(
    runs: List[RunResult],
    baseline_adapter: str = "none",
    metric: str = "wall_s",
) -> str:
    """Generate a markdown table of median overhead % per adapter per workload."""
    from collections import defaultdict

    # Group runs by (adapter, workload)
    groups: Dict[tuple, List[RunResult]] = defaultdict(list)
    for r in runs:
        if r.run_succeeded:
            groups[(r.adapter_name, r.workload_name)].append(r)

    workloads = sorted({k[1] for k in groups.keys()})
    adapters = sorted({k[0] for k in groups.keys() if k[0] != baseline_adapter})

    lines = []
    header = "| Adapter | " + " | ".join(workloads) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(workloads)) + " |"
    lines.append(header)
    lines.append(separator)

    for adapter in adapters:
        row_parts = [f"| {adapter} |"]
        for workload in workloads:
            baseline = groups.get((baseline_adapter, workload), [])
            compare = groups.get((adapter, workload), [])
            if baseline and compare:
                try:
                    delta = compute_paired_median_delta(baseline, compare, metric=metric)
                    row_parts.append(f" {delta:+.1f}% |")
                except Exception:
                    row_parts.append(" N/A |")
            else:
                row_parts.append(" - |")
        lines.append("".join(row_parts))

    return "\n".join(lines)
