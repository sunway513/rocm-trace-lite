"""profiler-bench CLI — three subcommands: verify, run, adapter-list.

Usage:
  profiler-bench verify [--threshold 5] [--level 1,2]
  profiler-bench run --config <yaml> [--rounds N] [--output <json>]
  profiler-bench adapter-list
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


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
        default=5.0,
        metavar="PCT",
        help="Overhead threshold %% (default: 5.0)",
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
    """Run regression gate for given levels and adapter."""
    from profiler_perf_bench.adapters import none as _none_mod  # noqa: F401
    from profiler_perf_bench.adapters import rtl as _rtl_mod  # noqa: F401
    from profiler_perf_bench.adapters.registry import global_registry
    from profiler_perf_bench.workloads.l1.gemm_hip import GemmHipSmall
    from profiler_perf_bench.workloads.l1.short_kernels_hip import ShortKernelsHip
    from profiler_perf_bench.workloads.l1.multi_stream_hip import MultiStreamHip
    from profiler_perf_bench.runner import BenchmarkRunner
    from profiler_perf_bench.report import check_regression, RegressionDetected

    # Default L1 workloads for verify
    l1_workloads = [GemmHipSmall(), ShortKernelsHip(), MultiStreamHip()]

    all_workloads = []
    for level in args.level:
        if level == 1:
            all_workloads.extend(l1_workloads)
        elif level == 2:
            print(
                "[SKIP] L2 workloads skipped: torch GPU not available on this host "
                "(gfx950 + torch 2.7.1+rocm6.2.4 known gap)"
            )
        else:
            print(f"[SKIP] Level {level} not supported in verify mode")

    if not all_workloads:
        print("No workloads to verify.")
        return 0

    try:
        baseline_cls = global_registry.get("none")
        adapter_cls = global_registry.get(args.adapter)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    regressions = []
    for workload in all_workloads:
        print(f"  Verifying {workload.name} with adapter '{args.adapter}' "
              f"(threshold={args.threshold:.1f}%, rounds={args.rounds})...")

        runner_none = BenchmarkRunner(baseline_cls(), workload.__class__(), args.rounds)
        runner_adapter = BenchmarkRunner(adapter_cls(), workload.__class__(), args.rounds)

        result_none = runner_none.run()
        result_adapter = runner_adapter.run()

        try:
            check_regression(
                result_none.rounds,
                result_adapter.rounds,
                threshold_pct=args.threshold,
                metric="wall_s",
            )
            overhead = _compute_overhead(result_none.rounds, result_adapter.rounds)
            print(f"    OK: overhead={overhead:.2f}%")
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

    # Workload registry
    workload_map = {
        "L1-gemm-small": GemmHipSmall,
        "L1-gemm-large": GemmHipLarge,
        "L1-short-kernels": ShortKernelsHip,
        "L1-multi-stream": MultiStreamHip,
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
