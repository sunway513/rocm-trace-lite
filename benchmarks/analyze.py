#!/usr/bin/env python3
"""Analyze RTL overhead benchmark results — generate reports and CI regression checks."""

import argparse
import csv
import io
import json
import sys
from pathlib import Path


def _load_results(path):
    return json.loads(Path(path).read_text())


def _format_markdown(results):
    lines = [
        "## RTL Overhead Benchmark Results",
        "",
        "| Workload | Profiler | Measured (s) | Per-iter (ms) | Overhead % | Process (s) | Startup OH | Trace (MB) | Verdict |",
        "|----------|----------|-------------|--------------|-----------|------------|-----------|-----------|---------|",
    ]

    for workload, profiler_map in results.items():
        baseline_sub = profiler_map.get("none", {}).get("median_subprocess_s", 0)

        for profiler, stats in profiler_map.items():
            median = stats["median_s"]
            per_iter = stats.get("per_iter_ms", 0.0)
            overhead = stats.get("overhead_pct")
            subprocess_s = stats.get("median_subprocess_s", 0)
            trace_mb = stats.get("median_trace_mb", 0)

            if overhead is None:
                oh_str = "-"
                verdict = "baseline"
            else:
                oh_str = f"{overhead:+.1f}%"
                if profiler == "rtl":
                    if overhead < 5:
                        verdict = "PASS"
                    elif overhead < 20:
                        verdict = "WARN"
                    else:
                        verdict = "FAIL"
                else:
                    verdict = "-"

            if profiler != "none" and baseline_sub > 0:
                startup_oh = subprocess_s - baseline_sub
                startup_str = f"{startup_oh:+.3f}s"
            else:
                startup_str = "-"

            trace_str = f"{trace_mb:.1f}" if trace_mb > 0 else "-"

            lines.append(
                f"| {workload} | {profiler} | {median:.4f} | {per_iter:.3f} | "
                f"{oh_str} | {subprocess_s:.3f} | {startup_str} | {trace_str} | {verdict} |"
            )
    return "\n".join(lines)


def _format_csv(results):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "workload", "profiler", "median_s", "per_iter_ms", "overhead_pct",
        "subprocess_s", "startup_oh_s", "trace_mb",
    ])
    for workload, profiler_map in results.items():
        baseline_sub = profiler_map.get("none", {}).get("median_subprocess_s", 0)
        for profiler, stats in profiler_map.items():
            startup_oh = ""
            if profiler != "none" and baseline_sub > 0:
                startup_oh = f"{stats.get('median_subprocess_s', 0) - baseline_sub:.4f}"
            writer.writerow([
                workload,
                profiler,
                f"{stats['median_s']:.4f}",
                f"{stats.get('per_iter_ms', 0.0):.4f}",
                f"{stats['overhead_pct']:.2f}" if stats.get("overhead_pct") is not None else "",
                f"{stats.get('median_subprocess_s', 0):.4f}",
                startup_oh,
                f"{stats.get('median_trace_mb', 0):.3f}",
            ])
    return buf.getvalue()


def _check_regression(results, threshold):
    failures = []
    for workload, profiler_map in results.items():
        rtl = profiler_map.get("rtl")
        if rtl is None:
            continue
        overhead = rtl.get("overhead_pct")
        if overhead is not None and overhead > threshold:
            failures.append((workload, overhead))
    return failures


def main():
    parser = argparse.ArgumentParser(description="Analyze RTL overhead benchmark results.")
    parser.add_argument("--input", default="results.json", help="Path to results JSON")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--check-regression", action="store_true", help="Exit 1 if rtl overhead exceeds threshold")
    parser.add_argument("--threshold", type=float, default=5.0, help="Max allowed overhead %% for regression check")
    args = parser.parse_args()

    results = _load_results(args.input)

    if args.format == "markdown":
        print(_format_markdown(results))
    else:
        print(_format_csv(results))

    if args.check_regression:
        failures = _check_regression(results, args.threshold)
        if failures:
            print(f"\nREGRESSION: rtl overhead exceeds {args.threshold}%:", file=sys.stderr)
            for workload, overhead in failures:
                print(f"  {workload}: {overhead:.1f}%", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"\nAll rtl overheads within {args.threshold}% threshold.", file=sys.stderr)


if __name__ == "__main__":
    main()
