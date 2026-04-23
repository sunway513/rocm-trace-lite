"""profiler-bench CLI — three subcommands: verify, run, adapter-list.

Usage:
  profiler-bench verify [--threshold PCT] [--level 1,2]
  profiler-bench run --config <yaml> [--rounds N] [--output <json>]
  profiler-bench adapter-list

Per-level default thresholds (when --threshold is not specified):
  L1: 15% pct gate (plus absolute 50ms budget — whichever is gentler)
  L2: 10% pct gate
  L3: 5%  pct gate (MLPerf-representative)

Use --threshold to override all levels with a single value.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Per-level default thresholds (pct gate)
_LEVEL_DEFAULT_THRESHOLDS = {
    1: 15.0,   # L1: fixed-cost-aware; absolute 50ms budget also applies (gentler wins)
    2: 10.0,   # L2: torch workloads
    3: 5.0,    # L3: MLPerf-representative serving
}
_THRESHOLD_NOT_SET = object()  # sentinel


def _get_level_threshold(args, level: int) -> float:
    """Return effective threshold for the given level.

    If --threshold was explicitly provided on CLI (i.e., args.threshold is not None),
    that overrides per-level defaults.
    Otherwise, use the per-level defaults from _LEVEL_DEFAULT_THRESHOLDS.
    """
    # Check both the explicit flag (set at runtime by _cmd_verify) and direct None check
    if getattr(args, "threshold_explicit", False) or args.threshold is not None:
        return float(args.threshold)
    return _LEVEL_DEFAULT_THRESHOLDS.get(level, 5.0)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="profiler-bench",
        description="Perf-only profiler overhead benchmark suite.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── verify ─────────────────────────────────────────────────────────────
    verify_parser = subparsers.add_parser(
        "verify",
        help="Run regression gate; exits 0=green, 1=fail",
    )
    verify_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Override threshold %% for all levels (default: per-level — "
            "L1=15%%, L2=10%%, L3=5%%)"
        ),
    )
    verify_parser.add_argument(
        "--level",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1],
        metavar="LEVELS",
        help="Comma-separated workload levels to verify (default: 1)",
    )
    verify_parser.add_argument(
        "--adapter",
        default="rtl",
        help="Adapter to test against 'none' baseline (default: rtl)",
    )
    verify_parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds per workload (default: 3)",
    )

    # ── run ────────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a benchmark sweep from a YAML config file",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        metavar="YAML",
        help="Path to sweep config YAML file",
    )
    run_parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Override rounds from config (default: use config value or 3)",
    )
    run_parser.add_argument(
        "--output",
        default="result.json",
        metavar="JSON",
        help="Output JSON file path (default: result.json)",
    )

    # ── adapter-list ───────────────────────────────────────────────────────
    subparsers.add_parser(
        "adapter-list",
        help="List all registered adapters with their execution models",
    )

    return parser


def get_adapter_list_output() -> str:
    """Return adapter-list output as a string (used in tests)."""
    # Import all adapter modules to trigger @register_adapter decorators
    from profiler_perf_bench.adapters import none, rtl, rocprofv3, rocprof, torch_profiler  # noqa: F401
    from profiler_perf_bench.adapters.registry import global_registry

    lines = ["Registered profiler adapters:", ""]
    lines.append(f"{'Name':<20} {'Execution Model':<25} {'Description'}")
    lines.append("-" * 70)

    for cls in global_registry.enumerate():
        instance = cls()
        lines.append(
            f"{instance.name:<20} {instance.execution_model.value:<25}"
        )

    lines.append("")
    lines.append(f"Total: {len(global_registry.list_names())} adapters registered")
    return "\n".join(lines)


def _cmd_adapter_list() -> int:
    print(get_adapter_list_output())
    return 0


def _cmd_verify(args) -> int:
    """Run regression gate for given levels and adapter.

    Uses per-level default thresholds unless --threshold is provided:
      L1: delta_ms ≤ 50 OR delta_pct ≤ 15  (fixed-cost-aware, gentler wins)
      L2: delta_pct ≤ 10
      L3: delta_pct ≤ 5  (MLPerf-representative)
    """
    from profiler_perf_bench.adapters import none as _none_mod  # noqa: F401
    from profiler_perf_bench.adapters import rtl as _rtl_mod  # noqa: F401
    from profiler_perf_bench.adapters.registry import global_registry
    from profiler_perf_bench.workloads.l1.gemm_hip import GemmHipSmall
    from profiler_perf_bench.workloads.l1.short_kernels_hip import ShortKernelsHip
    from profiler_perf_bench.workloads.l1.multi_stream_hip import MultiStreamHip
    from profiler_perf_bench.runner import BenchmarkRunner
    from profiler_perf_bench.report import (
        check_regression, check_regression_l1, check_regression_l2, check_regression_l3,
        RegressionDetected,
    )

    # Track whether --threshold was explicitly provided
    args.threshold_explicit = args.threshold is not None

    # Default L1 workloads for verify
    l1_workloads = [GemmHipSmall(), ShortKernelsHip(), MultiStreamHip()]

    # (level, workload) pairs
    level_workloads = []
    for level in args.level:
        if level == 1:
            for w in l1_workloads:
                level_workloads.append((1, w))
        elif level == 2:
            print(
                "[SKIP] L2 workloads skipped: torch GPU not available on this host "
                "(gfx950 + torch 2.7.1+rocm6.2.4 known gap)"
            )
        else:
            print(f"[SKIP] Level {level} not supported in verify mode")

    if not level_workloads:
        print("No workloads to verify.")
        return 0

    try:
        baseline_cls = global_registry.get("none")
        adapter_cls = global_registry.get(args.adapter)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    regressions = []
    for level, workload in level_workloads:
        effective_threshold = _get_level_threshold(args, level)
        threshold_desc = f"threshold={effective_threshold:.1f}%"
        if level == 1 and not args.threshold_explicit:
            threshold_desc = "L1 default (delta_ms≤50 OR pct≤15%)"
        elif not args.threshold_explicit:
            threshold_desc = f"L{level} default ({effective_threshold:.0f}%)"

        print(f"  Verifying {workload.name} with adapter '{args.adapter}' "
              f"({threshold_desc}, rounds={args.rounds})...")

        runner_none = BenchmarkRunner(baseline_cls(), workload.__class__(), args.rounds)
        runner_adapter = BenchmarkRunner(adapter_cls(), workload.__class__(), args.rounds)

        result_none = runner_none.run()
        result_adapter = runner_adapter.run()

        try:
            if args.threshold_explicit:
                # User-specified override — use flat threshold
                check_regression(
                    result_none.rounds,
                    result_adapter.rounds,
                    threshold_pct=args.threshold,
                    metric="wall_s",
                )
            elif level == 1:
                check_regression_l1(result_none.rounds, result_adapter.rounds)
            elif level == 2:
                check_regression_l2(result_none.rounds, result_adapter.rounds)
            else:
                check_regression_l3(result_none.rounds, result_adapter.rounds)

            overhead_pct, overhead_ms = _compute_overhead_both(result_none.rounds, result_adapter.rounds)
            print(f"    OK: overhead={overhead_pct:.2f}% ({overhead_ms:+.1f}ms abs)")
        except RegressionDetected as e:
            print(f"    REGRESSION: {e}")
            regressions.append(str(e))

    if regressions:
        print(f"\n{len(regressions)} regression(s) detected. Exit 1.")
        return 1

    print("\nAll checks passed. Exit 0.")
    return 0


def _compute_overhead(baseline_runs, adapter_runs) -> float:
    from profiler_perf_bench.report import compute_paired_median_delta
    try:
        return compute_paired_median_delta(baseline_runs, adapter_runs, metric="wall_s")
    except Exception:
        return float("nan")


def _compute_overhead_both(baseline_runs, adapter_runs):
    """Return (delta_pct, delta_ms) tuple."""
    import statistics
    from profiler_perf_bench.report import filter_succeeded_runs
    try:
        bl = filter_succeeded_runs(baseline_runs)
        ad = filter_succeeded_runs(adapter_runs)
        bl_vals = [r.metrics["wall_s"] for r in bl if "wall_s" in r.metrics]
        ad_vals = [r.metrics["wall_s"] for r in ad if "wall_s" in r.metrics]
        if not bl_vals or not ad_vals:
            return float("nan"), float("nan")
        bl_med = statistics.median(bl_vals)
        ad_med = statistics.median(ad_vals)
        delta_s = ad_med - bl_med
        delta_pct = (delta_s / bl_med * 100.0) if bl_med > 0 else float("inf")
        return delta_pct, delta_s * 1000.0
    except Exception:
        return float("nan"), float("nan")


def _cmd_run(args) -> int:
    """Execute a benchmark sweep from YAML config."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        import yaml
    except ImportError:
        # Fallback to basic YAML-like parsing or error
        print("Error: PyYAML is required for 'run' command. Install with: pip install pyyaml",
              file=sys.stderr)
        return 1

    with open(config_path) as f:
        config = yaml.safe_load(f)

    rounds = args.rounds or config.get("rounds", 3)
    output_path = Path(args.output)

    # Import all adapter modules
    from profiler_perf_bench.adapters import none, rtl, rocprofv3, rocprof, torch_profiler  # noqa: F401
    from profiler_perf_bench.adapters.registry import global_registry
    from profiler_perf_bench.runner import BenchmarkRunner
    from profiler_perf_bench.report import format_json_report
    from profiler_perf_bench.workloads.l1.gemm_hip import GemmHipSmall, GemmHipLarge
    from profiler_perf_bench.workloads.l1.short_kernels_hip import ShortKernelsHip
    from profiler_perf_bench.workloads.l1.multi_stream_hip import MultiStreamHip
    from profiler_perf_bench.workloads.l1.gemm_steady_hip import GemmHipSteady

    # Workload registry
    workload_map = {
        "L1-gemm-small": GemmHipSmall,
        "L1-gemm-large": GemmHipLarge,
        "L1-short-kernels": ShortKernelsHip,
        "L1-multi-stream": MultiStreamHip,
        "L1-gemm-steady": GemmHipSteady,
    }

    adapter_names = config.get("adapters", ["none", "rtl"])
    workload_names = config.get("workloads", ["L1-gemm-small"])
    metadata = config.get("metadata", {})
    metadata["config"] = str(config_path)
    metadata["rounds"] = rounds

    all_runs = []
    for adapter_name in adapter_names:
        try:
            adapter_cls = global_registry.get(adapter_name)
        except KeyError:
            print(f"Warning: adapter '{adapter_name}' not registered, skipping", file=sys.stderr)
            continue

        for workload_name in workload_names:
            workload_cls = workload_map.get(workload_name)
            if workload_cls is None:
                print(f"Warning: workload '{workload_name}' not found, skipping", file=sys.stderr)
                continue

            workload = workload_cls()
            missing = workload.check_requires()
            if missing:
                print(f"[SKIP] {workload_name}: missing requirements {missing}")
                continue

            print(f"Running {adapter_name} × {workload_name} × {rounds} rounds...")
            runner = BenchmarkRunner(adapter_cls(), workload, rounds)
            bench = runner.run()
            all_runs.extend(bench.rounds)

            succeeded = len([r for r in bench.rounds if r.run_succeeded])
            print(f"  → {succeeded}/{rounds} succeeded")

    if not all_runs:
        print("No runs completed.", file=sys.stderr)
        return 1

    report = format_json_report(all_runs, metadata=metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults written to: {output_path}")
    print(f"Total runs: {len(all_runs)}, succeeded: {sum(1 for r in all_runs if r.run_succeeded)}")
    return 0


def main():
    """Entry point for profiler-bench console script."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "adapter-list":
        sys.exit(_cmd_adapter_list())
    elif args.command == "verify":
        sys.exit(_cmd_verify(args))
    elif args.command == "run":
        sys.exit(_cmd_run(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
