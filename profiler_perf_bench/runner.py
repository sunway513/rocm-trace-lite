"""BenchmarkRunner: adapter × workload → RunResult / BenchResult.

Handles:
  - external_wrapper: Popen with modified cmd + env
  - in_process_python: start/stop around in-process workload
  - Sanity gate application
  - Wall time and RSS measurement
  - tmpdir management per run
"""

from __future__ import annotations
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .adapters.base import ExecutionModel, ProfilerAdapter
from .metrics import BenchResult, RunResult
from .sanity import check_sanity
from .workloads.base import Workload


def _run_measured(cmd, env, cwd):
    """Measure one workload via a fresh supervisor, excluding previous rounds.

    A fresh supervisor also avoids the fork-before-exec RSS floor of a runner
    that has already imported torch. Its launch cost applies to every adapter.
    """
    import json
    import sys
    report = Path(cwd) / 'child-usage.json'
    helper = str(Path(__file__).with_name('_measure_child.py'))
    proc = subprocess.run([sys.executable, helper, str(report), *cmd],
                          env=env, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        return proc, None
    usage = json.loads(report.read_text())
    return subprocess.CompletedProcess(cmd, usage['returncode'], proc.stdout, proc.stderr), usage['peak_rss_MB']

class BenchmarkRunner:
    """Runs (adapter × workload) N rounds and returns aggregated BenchResult."""

    def __init__(
        self,
        adapter: ProfilerAdapter,
        workload: Workload,
        rounds: int = 3,
    ):
        self.adapter = adapter
        self.workload = workload
        self.rounds = rounds

    def run_once(self) -> RunResult:
        """Execute a single round. Returns RunResult with metrics + sanity gate result."""
        with tempfile.TemporaryDirectory(prefix="ppb_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            return self._run_once_in(tmpdir)

    def _run_once_in(self, tmpdir: Path) -> RunResult:
        adapter = self.adapter
        workload = self.workload

        # Build base command + env from workload
        base_cmd = workload.cmd()
        base_env = {**os.environ, **workload.env()}

        exit_code = -1
        stdout_str = ""
        stderr_str = ""
        wall_start = time.perf_counter()
        subprocess_wall = 0.0
        peak_rss_mb = 0.0
        trace_bytes = 0

        if adapter.execution_model == ExecutionModel.EXTERNAL_WRAPPER:
            # Apply adapter's cmd/env modifications
            try:
                modified_cmd, modified_env = adapter.prepare_run(base_cmd, base_env, tmpdir)
            except ValueError as e:
                return RunResult(adapter_name=adapter.name, workload_name=workload.name,
                                 round_idx=0, metrics=_empty_metrics(False), run_succeeded=False,
                                 dropped_reason=f'adapter_prepare_failed: {e}')

            sub_start = time.perf_counter()
            try:
                proc, peak_rss_mb = _run_measured(modified_cmd, modified_env, str(tmpdir))
                exit_code = proc.returncode
                stdout_str = proc.stdout
                stderr_str = proc.stderr
            except FileNotFoundError as e:
                # Binary not found — treat as crashed
                exit_code = -2
                stderr_str = str(e)
            sub_end = time.perf_counter()
            subprocess_wall = sub_end - sub_start

        elif adapter.execution_model == ExecutionModel.IN_PROCESS_PYTHON:
            return RunResult(adapter_name=adapter.name, workload_name=workload.name,
                             round_idx=0, metrics=_empty_metrics(False), run_succeeded=False,
                             dropped_reason='in_process_adapter_requires_workload_bootstrap')
        wall_end = time.perf_counter()
        wall_s = wall_end - wall_start

        # Measure trace artifacts size
        artifact_glob = adapter.artifact_glob()
        if artifact_glob:
            import glob as _glob
            matches = _glob.glob(str(tmpdir / "**" / artifact_glob), recursive=True)
            if not matches:
                matches = _glob.glob(str(tmpdir / artifact_glob))
            trace_bytes = sum(
                Path(m).stat().st_size for m in matches if Path(m).is_file()
            )

        # Parse workload-specific metrics
        try:
            workload_metrics = workload.parse_metrics(stdout_str, stderr_str, tmpdir)
        except Exception:
            workload_metrics = {}

        # Extract L3 successful_requests if present
        l3_successful_requests = workload_metrics.get("successful_requests", None)

        # Run sanity checks
        sanity = check_sanity(
            exit_code=exit_code,
            adapter_name=adapter.name,
            workload_level=workload.level,
            artifact_dir=tmpdir,
            artifact_glob=artifact_glob,
            metrics={},
            l3_successful_requests=l3_successful_requests,
        )

        metrics: dict = {
            "wall_s": wall_s,
            "subprocess_s": subprocess_wall,
            "adapter_init_s": None,
            "adapter_shutdown_s": None,
            "trace_bytes": trace_bytes,
            "peak_rss_MB": peak_rss_mb,
            "run_succeeded": sanity.run_succeeded,
            "dropped_reason": sanity.dropped_reason,
        }
        metrics.update(workload_metrics)

        return RunResult(
            adapter_name=adapter.name,
            workload_name=workload.name,
            round_idx=0,  # updated in run()
            metrics=metrics,
            run_succeeded=sanity.run_succeeded,
            dropped_reason=sanity.dropped_reason,
        )

    def run(self) -> BenchResult:
        """Run self.rounds rounds and return BenchResult aggregate."""
        bench = BenchResult(
            adapter_name=self.adapter.name,
            workload_name=self.workload.name,
        )
        for i in range(self.rounds):
            result = self.run_once()
            result.round_idx = i
            bench.rounds.append(result)
        return bench


def _empty_metrics(run_succeeded: bool = False) -> dict:
    return {
        "wall_s": 0.0,
        "subprocess_s": 0.0,
        "adapter_init_s": None,
        "adapter_shutdown_s": None,
        "trace_bytes": 0,
        "peak_rss_MB": 0.0,
        "run_succeeded": run_succeeded,
        "dropped_reason": None,
    }
