#!/usr/bin/env python3
"""RTL overhead benchmark — compare profiling tools.

Runs a workload script under different profiler configurations and measures
wall-clock overhead, startup/shutdown latency, and trace file sizes.
"""

import argparse
import glob
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Workload and profiler registries
# ---------------------------------------------------------------------------

_BENCH_DIR = Path(__file__).parent

WORKLOADS = {
    "gemm": str(_BENCH_DIR / "workloads" / "gemm.py"),
    "short_kernels": str(_BENCH_DIR / "workloads" / "short_kernels.py"),
    "mixed_model": str(_BENCH_DIR / "workloads" / "mixed_model.py"),
    "gemm_prod": str(_BENCH_DIR / "workloads" / "gemm_prod.py"),
    "inference_sim": str(_BENCH_DIR / "workloads" / "inference_sim.py"),
    "nccl_comm": str(_BENCH_DIR / "workloads" / "nccl_comm.py"),
}

# Workloads that require torchrun (multi-GPU)
_DISTRIBUTED_WORKLOADS = {"nccl_comm"}

PROFILERS = ["none", "rtl", "rtl_hip", "rocprofv3", "roctracer"]


# ---------------------------------------------------------------------------
# librtl.so discovery
# ---------------------------------------------------------------------------

def _find_librtl() -> Optional[str]:
    """Return path to librtl.so, or None if not found."""
    try:
        from rocm_trace_lite import get_lib_path  # type: ignore
        return get_lib_path()
    except Exception:
        pass
    repo_root_lib = _BENCH_DIR.parent / "librtl.so"
    if repo_root_lib.is_file():
        return str(repo_root_lib)
    sys_lib = Path("/usr/local/lib/librtl.so")
    if sys_lib.is_file():
        return str(sys_lib)
    return None


# ---------------------------------------------------------------------------
# Trace file size measurement
# ---------------------------------------------------------------------------

def _measure_trace_files(tmpdir: Path) -> Dict:
    """Scan tmpdir for profiler output files, return total size and file list."""
    trace_files = []
    total_bytes = 0
    for f in tmpdir.rglob("*"):
        if f.is_file():
            sz = f.stat().st_size
            trace_files.append({"name": f.name, "bytes": sz})
            total_bytes += sz
    return {"total_bytes": total_bytes, "total_mb": round(total_bytes / 1024 / 1024, 3), "files": trace_files}


# ---------------------------------------------------------------------------
# Per-profiler command builders
# ---------------------------------------------------------------------------

def _torchrun_cmd(nproc: int, script: str, args: List[str]) -> List[str]:
    """Build torchrun command with a random master port to avoid conflicts."""
    port = random.randint(29500, 39999)
    return ["torchrun", "--nproc_per_node", str(nproc), "--master_port", str(port), script] + args


def _add_distributed_env(env: dict, nproc: int) -> dict:
    """Add env vars required for RCCL multi-GPU."""
    if nproc > 1:
        env["HSA_NO_SCRATCH_RECLAIM"] = "1"
    return env


def _run_none(workload_script: str, workload_args: List[str], tmpdir: Path, nproc: int):
    """Baseline: run workload directly, no profiler."""
    if nproc > 1:
        cmd = _torchrun_cmd(nproc, workload_script, workload_args)
    else:
        cmd = [sys.executable, workload_script] + workload_args
    env = _add_distributed_env(os.environ.copy(), nproc)
    return cmd, env


def _run_rtl(workload_script: str, workload_args: List[str], tmpdir: Path, nproc: int):
    """RTL: inject HSA_TOOLS_LIB + RTL_OUTPUT into the environment."""
    lib = _find_librtl()
    if lib is None:
        raise RuntimeError(
            "librtl.so not found. Build the library first (make) or install the wheel."
        )
    if nproc > 1:
        cmd = _torchrun_cmd(nproc, workload_script, workload_args)
    else:
        cmd = [sys.executable, workload_script] + workload_args
    env = _add_distributed_env(os.environ.copy(), nproc)
    env["HSA_TOOLS_LIB"] = lib
    env["RTL_OUTPUT"] = str(tmpdir / "trace_%p.db")
    return cmd, env


def _run_rtl_hip(workload_script: str, workload_args: List[str], tmpdir: Path, nproc: int):
    """RTL HIP mode: CLR profiler path instead of HSA signal injection.

    Requires a custom-built libamdhip64.so with hipProfiler*Ext symbols.
    Uses LD_PRELOAD to override the system library at runtime without
    replacing it (which would break PyTorch's eager symbol resolution).
    """
    lib = _find_librtl()
    if lib is None:
        raise RuntimeError(
            "librtl.so not found. Build the library first (make) or install the wheel."
        )
    # Find the custom-built libamdhip64.so with profiler extensions
    hip_lib = "/opt/rocm/lib/libamdhip64.so.7.2.0-8e637b7173"
    if not os.path.isfile(hip_lib):
        raise RuntimeError(
            f"Custom libamdhip64.so not found at {hip_lib}. "
            "Build from rocm-systems ROCM-1667-12 branch first."
        )
    if nproc > 1:
        cmd = _torchrun_cmd(nproc, workload_script, workload_args)
    else:
        cmd = [sys.executable, workload_script] + workload_args
    env = _add_distributed_env(os.environ.copy(), nproc)
    env["HSA_TOOLS_LIB"] = lib
    env["RTL_OUTPUT"] = str(tmpdir / "trace_%p.db")
    env["RTL_MODE"] = "hip"
    env["GPU_CLR_PROFILE_OUTPUT"] = "/dev/null"
    # LD_PRELOAD the custom library + stubs for missing ROCR symbols.
    # The stubs satisfy hsa_ext_image_* and hsa_amd_memory_async_batch_copy
    # which exist in newer ROCR but not in ROCm 7.2. These are only called
    # on image API paths, never during compute workloads.
    stubs_lib = "/opt/rocm/lib/libhsa_stubs.so"
    preload = hip_lib
    if os.path.isfile(stubs_lib):
        preload = f"{stubs_lib} {hip_lib}"
    env["LD_PRELOAD"] = preload
    return cmd, env


def _run_rocprofv3(workload_script: str, workload_args: List[str], tmpdir: Path, nproc: int):
    """rocprofv3: use --runtime-trace flag."""
    if shutil.which("rocprofv3") is None:
        raise RuntimeError("rocprofv3 not found in PATH")
    if nproc > 1:
        app_cmd = _torchrun_cmd(nproc, workload_script, workload_args)
    else:
        app_cmd = [sys.executable, workload_script] + workload_args
    cmd = [
        "rocprofv3",
        "--runtime-trace",
        "-o", str(tmpdir / "rocprof"),
        "--",
    ] + app_cmd
    env = _add_distributed_env(os.environ.copy(), nproc)
    return cmd, env


def _run_roctracer(workload_script: str, workload_args: List[str], tmpdir: Path, nproc: int):
    """roctracer (legacy rocprof): use --hip-trace flag."""
    if shutil.which("rocprof") is None:
        raise RuntimeError("rocprof not found in PATH")
    if nproc > 1:
        app_cmd = _torchrun_cmd(nproc, workload_script, workload_args)
    else:
        app_cmd = [sys.executable, workload_script] + workload_args
    cmd = [
        "rocprof",
        "--hip-trace",
        "-o", str(tmpdir / "roctracer.csv"),
    ] + app_cmd
    env = _add_distributed_env(os.environ.copy(), nproc)
    return cmd, env


_BUILDERS = {
    "none": _run_none,
    "rtl": _run_rtl,
    "rtl_hip": _run_rtl_hip,
    "rocprofv3": _run_rocprofv3,
    "roctracer": _run_roctracer,
}


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    workload: str,
    profiler: str,
    runs: int = 5,
    workload_args: Optional[List[str]] = None,
    nproc: int = 1,
) -> Dict:
    """Run *workload* under *profiler* for *runs* repetitions.

    Returns a dict with timing statistics, startup/shutdown overhead, and trace file sizes.
    """
    workload_args = workload_args or []
    script = WORKLOADS[workload]
    builder = _BUILDERS[profiler]

    wall_times: List[float] = []
    per_iter_ms_values: List[float] = []
    subprocess_times: List[float] = []
    total_process_times: List[float] = []
    trace_sizes: List[Dict] = []

    for i in range(runs):
        tmpdir = Path(tempfile.mkdtemp(prefix=f"rtl_bench_{profiler}_{i}_"))
        try:
            cmd, env = builder(script, workload_args, tmpdir, nproc)

            # Measure full subprocess wall time (includes Python startup, import, CUDA init, profiler init/shutdown)
            t_sub_start = time.perf_counter()
            result = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            t_sub_end = time.perf_counter()
            subprocess_wall_s = t_sub_end - t_sub_start
            subprocess_times.append(subprocess_wall_s)

            # Parse JSON from stdout
            output = result.stdout.decode("utf-8", errors="replace").strip()
            json_line = None
            for line in reversed(output.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    json_line = line
                    break
            if json_line is None:
                raise RuntimeError(
                    f"No JSON output from workload (stdout):\n{output[:500]}"
                )
            data = json.loads(json_line)
            wall_times.append(float(data["wall_s"]))
            per_iter_ms_values.append(float(data.get("per_iter_ms", 0.0)))
            total_process_times.append(float(data.get("total_process_s", 0.0)))

            # Measure trace output file sizes
            trace_info = _measure_trace_files(tmpdir)
            trace_sizes.append(trace_info)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    mean_s = statistics.mean(wall_times)
    median_s = statistics.median(wall_times)
    stdev_s = statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0
    per_iter_ms = statistics.median(per_iter_ms_values) if per_iter_ms_values else 0.0

    median_subprocess_s = statistics.median(subprocess_times)
    median_total_process_s = statistics.median(total_process_times) if any(total_process_times) else 0.0

    # Startup overhead = subprocess_time - total_process_time (profiler load time + extra process overhead)
    # Shutdown overhead = total_process_time - (warmup + measured region)
    # These are approximations; the key comparison is between profiler vs baseline subprocess times

    median_trace_mb = statistics.median([t["total_mb"] for t in trace_sizes]) if trace_sizes else 0.0

    return {
        "profiler": profiler,
        "workload": workload,
        "nproc": nproc,
        "runs": wall_times,
        "mean_s": mean_s,
        "median_s": median_s,
        "stdev_s": stdev_s,
        "per_iter_ms": per_iter_ms,
        "subprocess_times": subprocess_times,
        "median_subprocess_s": median_subprocess_s,
        "median_total_process_s": median_total_process_s,
        "median_trace_mb": median_trace_mb,
        "trace_sizes": trace_sizes,
    }


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------

def _print_table(results: dict):
    """Print a human-readable summary table to stderr."""
    header = (
        f"{'Workload':<16} | {'Profiler':<12} | {'Measured(s)':>11} | "
        f"{'Overhead%':>9} | {'Process(s)':>10} | {'Startup OH':>10} | {'Trace(MB)':>9}"
    )
    sep = "-" * len(header)
    print(header, file=sys.stderr)
    print(sep, file=sys.stderr)

    for workload_name, profiler_map in results.items():
        baseline_subprocess = None
        if "none" in profiler_map:
            baseline_subprocess = profiler_map["none"]["median_subprocess_s"]

        for profiler_name, stats in profiler_map.items():
            median_s = stats["median_s"]
            overhead = stats.get("overhead_pct")
            subprocess_s = stats["median_subprocess_s"]
            trace_mb = stats["median_trace_mb"]

            oh_str = "-" if overhead is None else f"{overhead:+.1f}%"

            # Startup overhead = how much longer the full process takes compared to baseline
            if baseline_subprocess is not None and profiler_name != "none":
                startup_oh_s = subprocess_s - baseline_subprocess
                startup_str = f"{startup_oh_s:+.3f}s"
            else:
                startup_str = "-"

            trace_str = f"{trace_mb:.1f}" if trace_mb > 0 else "-"

            print(
                f"{workload_name:<16} | {profiler_name:<12} | {median_s:>11.4f} | "
                f"{oh_str:>9} | {subprocess_s:>10.3f} | {startup_str:>10} | {trace_str:>9}",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RTL overhead benchmark — compare profiling tools."
    )
    parser.add_argument(
        "--workload",
        choices=list(WORKLOADS.keys()) + ["all"],
        default="all",
        help="Which workload to run (default: all)",
    )
    parser.add_argument(
        "--profiler",
        choices=PROFILERS + ["all"],
        default="all",
        help="Which profiler configuration to use (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of repeated runs per (workload, profiler) combination (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to save JSON results (default: results.json)",
    )
    parser.add_argument(
        "--workload-args",
        default="",
        help="Extra arguments to pass through to the workload script (as a single quoted string)",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of GPUs for distributed workloads (default: 1)",
    )
    args = parser.parse_args()

    workloads_to_run = list(WORKLOADS.keys()) if args.workload == "all" else [args.workload]
    profilers_to_run = PROFILERS if args.profiler == "all" else [args.profiler]

    extra_args: List[str] = args.workload_args.split() if args.workload_args else []

    # results[workload][profiler] = stats dict
    results: Dict[str, Dict[str, Dict]] = {}

    for workload in workloads_to_run:
        script_path = WORKLOADS[workload]
        if not Path(script_path).is_file():
            print(
                f"[skip] Workload '{workload}' script not found: {script_path}",
                file=sys.stderr,
            )
            continue

        # Determine nproc: distributed workloads use --nproc, others use 1
        nproc = args.nproc if workload in _DISTRIBUTED_WORKLOADS else 1
        if workload in _DISTRIBUTED_WORKLOADS and nproc < 2:
            print(
                f"[skip] Workload '{workload}' requires --nproc >= 2 (got {nproc})",
                file=sys.stderr,
            )
            continue

        results.setdefault(workload, {})

        for profiler in profilers_to_run:
            print(
                f"[run ] workload={workload}  profiler={profiler}  runs={args.runs}  nproc={nproc}",
                file=sys.stderr,
            )
            try:
                stats = run_benchmark(
                    workload=workload,
                    profiler=profiler,
                    runs=args.runs,
                    workload_args=extra_args,
                    nproc=nproc,
                )
                results[workload][profiler] = stats
            except RuntimeError as exc:
                print(f"[warn] Skipping {profiler}: {exc}", file=sys.stderr)
            except subprocess.CalledProcessError as exc:
                stderr_out = exc.stderr.decode("utf-8", errors="replace")[:500] if exc.stderr else ""
                print(
                    f"[warn] Workload '{workload}' under '{profiler}' failed "
                    f"(exit {exc.returncode}):\n{stderr_out}",
                    file=sys.stderr,
                )

    # Compute overhead relative to baseline ("none")
    for workload_name, profiler_map in results.items():
        baseline_stats = profiler_map.get("none")
        if baseline_stats is None:
            continue
        baseline = baseline_stats["median_s"]
        if baseline <= 0:
            continue
        for profiler_name, stats in profiler_map.items():
            if profiler_name == "none":
                continue
            profiled = stats["median_s"]
            stats["overhead_pct"] = (profiled - baseline) / baseline * 100.0

    # Print summary table
    if results:
        print("", file=sys.stderr)
        _print_table(results)

    # Save JSON results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] Results saved to {output_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
