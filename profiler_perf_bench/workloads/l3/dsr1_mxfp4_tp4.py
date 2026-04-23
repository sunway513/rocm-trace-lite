"""L3 DeepSeek-R1 MXFP4 TP=4 workload.

Per spec §4.3:
  L3-dsr1-mxfp4-tp4: amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4, TP=4, fp8 KV,
  ISL=1024 OSL=1024 conc=4 (matches 2026-04-20 PR#94 validation run).

This class is unit-tested for config correctness but integration is skipped
per task spec (L3 = opt-in, not run in this task).
"""

import sys
from pathlib import Path
from typing import Optional, Callable, List

from ..base import Level, Workload

# Model identifier matching ATOM/models.json
_MODEL_ID = "amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
_TP = 4
_ISL = 1024
_OSL = 1024
_CONCURRENCY = 4


class DSR1MxFP4TP4Workload(Workload):
    """L3-dsr1-mxfp4-tp4: DeepSeek-R1 MxFP4 serving workload."""

    name = "L3-dsr1-mxfp4-tp4"
    level = Level.L3
    requires = [
        "python3",
        # vLLM/ATOM server entry point (checked at runtime, not import time)
    ]

    def cmd(self) -> List[str]:
        """Server launch command (ATOM/vLLM openai_server)."""
        return [
            sys.executable,
            "/app/ATOM/atom/entrypoints/openai_server.py",
            "--model", _MODEL_ID,
            "--tensor-parallel-size", str(_TP),
            "--kv-cache-dtype", "fp8",
            "--max-model-len", str(_ISL + _OSL),
            "--enforce-eager",
        ]

    def env(self) -> dict:
        return {}

    def ready_probe(self) -> Optional[Callable[[], bool]]:
        """Poll /health endpoint to detect server readiness."""
        import time
        import urllib.request

        def _probe():
            try:
                resp = urllib.request.urlopen(
                    "http://localhost:8000/health", timeout=5
                )
                return resp.status == 200
            except Exception:
                return False

        return _probe

    def client_cmd(self) -> Optional[List[str]]:
        return [
            sys.executable, "-m", "atom.benchmarks.benchmark_serving",
            "--backend", "openai",
            "--base-url", "http://localhost:8000",
            "--model", _MODEL_ID,
            "--num-prompts", "50",
            "--input-len", str(_ISL),
            "--output-len", str(_OSL),
            "--concurrency", str(_CONCURRENCY),
        ]

    def parse_metrics(self, stdout: str, stderr: str, artifact_dir: Path) -> dict:
        """Parse benchmark_serving output for L3 metrics."""
        metrics = {}
        for line in stdout.splitlines():
            # Try to parse key=value or "Key: value" patterns from benchmark output
            if "ttft" in line.lower() and "mean" in line.lower():
                try:
                    val = float(line.split()[-1])
                    metrics["ttft_ms_mean"] = val
                except (ValueError, IndexError):
                    pass
            if "successful requests" in line.lower():
                try:
                    val = int(line.split()[-1])
                    metrics["successful_requests"] = val
                except (ValueError, IndexError):
                    pass
        return metrics
