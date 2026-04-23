"""L3 GPT-OSS TP=1 workload.

Per spec §4.3: L3-gpt-oss-tp1, openai/gpt-oss-120b, TP=1, fp8 KV.
Note: documents the ATOM block_tables pre-existing bug; runs only if workaround env present.

Integration is skipped per task spec (L3 = opt-in).
"""

import sys
from pathlib import Path
from typing import Optional, Callable, List

from ..base import Level, Workload

_MODEL_ID = "openai/gpt-oss-120b"


class GptOssTP1Workload(Workload):
    """L3-gpt-oss-tp1: GPT-OSS TP=1 serving workload."""

    name = "L3-gpt-oss-tp1"
    level = Level.L3
    requires = ["python3"]

    def cmd(self) -> List[str]:
        return [
            sys.executable,
            "/app/ATOM/atom/entrypoints/openai_server.py",
            "--model", _MODEL_ID,
            "--tensor-parallel-size", "1",
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
