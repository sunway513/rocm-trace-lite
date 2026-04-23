"""Report generation: JSON + Markdown output, regression check, compare logic.

Per spec §3.5:
  - compare() runs a sweep with interleaved per-round branch order.
  - check_regression() raises RegressionDetected if any cell exceeds threshold_pct.

Per-level threshold defaults (PR#96 overhead framing correction):
  L1: delta_ms ≤ 50 OR delta_pct ≤ 15  (whichever is gentler — fixed-cost-aware)
  L2: delta_pct ≤ 10
  L3: delta_pct ≤ 5  (MLPerf-representative)

Fixed startup cost context:
  RTL adds ~25-40ms one-time init (HSA_TOOLS_LIB load + signal pool + SQLite schema).
  On 250ms microbench this appears as 10-17%; on 10s+ production runs it's 0.24-0.4%.
  Per-kernel cost is ≤1 µs/dispatch in lite mode.
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


def classify_overhead(delta_ms: float, baseline_ms: float) -> str:
    """Classify overhead as fixed_cost_dominated, workload_dominated, or mixed.

    Rules (per PR#96 correction spec §3):
      - "fixed_cost_dominated"  if delta_ms < 50 AND baseline_ms < 1000
      - "workload_dominated"    if baseline_ms > 3000
      - "mixed"                 otherwise

    This classification contextualises why short-run overhead % looks high:
    RTL fixed startup cost ≈ 25-40ms is amortised over longer runs.
    """
    if baseline_ms > 3000.0:
        return "workload_dominated"
    if delta_ms < 50.0 and baseline_ms < 1000.0:
        return "fixed_cost_dominated"
    return "mixed"


def format_json_report_with_deltas(
    runs: List[RunResult],
    baseline_adapter: str = "none",
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """Format runs as JSON report with per-adapter-per-workload delta_ms, delta_pct, classification.

    Extended version of format_json_report() — summary is a list (not dict) so each entry
    carries: adapter_name, workload_name, baseline_ms, adapter_ms, delta_ms, delta_pct, classification.
    """
    from collections import defaultdict

    results_list = [r.to_dict() for r in runs]

    # Group runs by (adapter, workload)
    groups: Dict[tuple, List[RunResult]] = defaultdict(list)
    for r in runs:
        if r.run_succeeded:
            groups[(r.adapter_name, r.workload_name)].append(r)

    workloads = sorted({k[1] for k in groups.keys()})
    adapters = sorted({k[0] for k in groups.keys()})

    summary_list = []
    for adapter in adapters:
        for workload in workloads:
            grp = groups.get((adapter, workload), [])
            if not grp:
                continue

            baseline_grp = groups.get((baseline_adapter, workload), [])
            adapter_vals = [r.metrics["wall_s"] for r in grp if "wall_s" in r.metrics]
            baseline_vals = [r.metrics["wall_s"] for r in baseline_grp if "wall_s" in r.metrics]

            if not adapter_vals:
                continue

            adapter_median_s = statistics.median(adapter_vals)
            entry: Dict[str, Any] = {
                "adapter_name": adapter,
                "workload_name": workload,
                "adapter_ms": round(adapter_median_s * 1000, 3),
            }

            if baseline_vals and adapter != baseline_adapter:
                baseline_median_s = statistics.median(baseline_vals)
                delta_s = adapter_median_s - baseline_median_s
                delta_ms = delta_s * 1000.0
                baseline_ms = baseline_median_s * 1000.0
                delta_pct = (delta_s / baseline_median_s * 100.0) if baseline_median_s > 0 else float("inf")

                entry["baseline_ms"] = round(baseline_ms, 3)
                entry["delta_ms"] = round(delta_ms, 3)
                entry["delta_pct"] = round(delta_pct, 3)
                entry["classification"] = classify_overhead(delta_ms, baseline_ms)
            elif adapter == baseline_adapter and baseline_vals:
                baseline_median_s = statistics.median(baseline_vals)
                entry["baseline_ms"] = round(baseline_median_s * 1000, 3)
                entry["delta_ms"] = 0.0
                entry["delta_pct"] = 0.0
                entry["classification"] = "baseline"

            summary_list.append(entry)

    return {
        "metadata": metadata or {},
        "results": results_list,
        "summary": summary_list,
    }


def check_regression_l1(
    baseline_runs: List[RunResult],
    adapter_runs: List[RunResult],
    metric: str = "wall_s",
) -> None:
    """L1 regression check: passes if delta_ms ≤ 50 OR delta_pct ≤ 15 (gentler wins).

    This is fixed-cost-aware: RTL startup adds ~25-40ms regardless of workload length.
    On short (<1s) L1 microbenchmarks, the absolute budget (50ms) is the meaningful gate.
    """
    baseline_ok = filter_succeeded_runs(baseline_runs)
    adapter_ok = filter_succeeded_runs(adapter_runs)

    baseline_vals = [r.metrics[metric] for r in baseline_ok if metric in r.metrics]
    adapter_vals = [r.metrics[metric] for r in adapter_ok if metric in r.metrics]

    if not baseline_vals or not adapter_vals:
        raise ValueError(f"No valid runs to compare on metric '{metric}'")

    baseline_median = statistics.median(baseline_vals)
    adapter_median = statistics.median(adapter_vals)

    delta_s = adapter_median - baseline_median
    delta_ms = delta_s * 1000.0
    delta_pct = (delta_s / baseline_median * 100.0) if baseline_median > 0 else float("inf")

    # Gentler gate: passes if EITHER condition holds
    passes_abs = delta_ms <= 50.0
    passes_pct = delta_pct <= 15.0

    if not (passes_abs or passes_pct):
        adapter_name = adapter_runs[0].adapter_name if adapter_runs else "unknown"
        workload_name = baseline_runs[0].workload_name if baseline_runs else "unknown"
        raise RegressionDetected(
            f"L1 regression: {adapter_name} vs none on {workload_name}/{metric}: "
            f"delta_ms={delta_ms:.1f} (budget ≤50ms) AND delta_pct={delta_pct:.1f}% (budget ≤15%) — "
            f"both exceeded"
        )


def check_regression_l2(
    baseline_runs: List[RunResult],
    adapter_runs: List[RunResult],
    metric: str = "wall_s",
) -> None:
    """L2 regression check: delta_pct ≤ 10%."""
    delta = compute_paired_median_delta(baseline_runs, adapter_runs, metric=metric)
    if delta > 10.0:
        adapter_name = adapter_runs[0].adapter_name if adapter_runs else "unknown"
        workload_name = baseline_runs[0].workload_name if baseline_runs else "unknown"
        raise RegressionDetected(
            f"L2 regression: {adapter_name} vs none on {workload_name}/{metric}: "
            f"{delta:.2f}% > 10.0% threshold"
        )


def check_regression_l3(
    baseline_runs: List[RunResult],
    adapter_runs: List[RunResult],
    metric: str = "wall_s",
) -> None:
    """L3 regression check: delta_pct ≤ 5% (MLPerf-representative)."""
    delta = compute_paired_median_delta(baseline_runs, adapter_runs, metric=metric)
    if delta > 5.0:
        adapter_name = adapter_runs[0].adapter_name if adapter_runs else "unknown"
        workload_name = baseline_runs[0].workload_name if baseline_runs else "unknown"
        raise RegressionDetected(
            f"L3 regression: {adapter_name} vs none on {workload_name}/{metric}: "
            f"{delta:.2f}% > 5.0% threshold"
        )


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
