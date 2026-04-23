"""L3 GLM-5 FP8 TP=8 workload.

Per spec §4.3: L3-glm5-fp8-tp8, zai-org/GLM-5-FP8, TP=8.
Skipped on single-GPU hosts per spec.

Integration is skipped per task spec (L3 = opt-in).
"""

import sys
from pathlib import Path
from typing import Optional, Callable, List

from ..base import Level, Workload

_MODEL_ID = "zai-org/GLM-5-FP8"


class GLM5FP8TP8Workload(Workload):
    """L3-glm5-fp8-tp8: GLM-5 FP8 TP=8 serving workload."""

    name = "L3-glm5-fp8-tp8"
    level = Level.L3
    requires = ["python3"]

    def cmd(self) -> List[str]:
        return [
            sys.executable,
            "/app/ATOM/atom/entrypoints/openai_server.py",
            "--model", _MODEL_ID,
            "--tensor-parallel-size", "8",
            "--kv-cache-dtype", "fp8",
        ]

    def env(self) -> dict:
        return {}

    def ready_probe(self) -> Optional[Callable[[], bool]]:
        import urllib.request

        def _probe():
            try:
                resp = urllib.request.urlopen("http://localhost:8000/health", timeout=5)
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
            "--num-prompts", "20",
        ]

    def parse_metrics(self, stdout: str, stderr: str, artifact_dir: Path) -> dict:
        return {}
