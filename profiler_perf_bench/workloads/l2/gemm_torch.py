"""L2 GEMM torch workload.

Per spec §4.2: L2-gemm-torch, uses benchmarks/workloads/gemm.py
"""

import sys
from pathlib import Path
from typing import Optional, Callable, List

from ..base import Level, Workload

# Path to the existing benchmarks/workloads/gemm.py
_GEMM_SCRIPT = str(
    Path(__file__).parent.parent.parent.parent / "benchmarks" / "workloads" / "gemm.py"
)


class GemmTorchWorkload(Workload):
    """L2-gemm-torch: torch-based GEMM workload."""

    name = "L2-gemm-torch"
    level = Level.L2
    requires = ["torch", _GEMM_SCRIPT]

    def cmd(self) -> List[str]:
        return [sys.executable, _GEMM_SCRIPT]

    def env(self) -> dict:
        return {}

    def ready_probe(self) -> Optional[Callable[[], bool]]:
        return None

    def client_cmd(self) -> Optional[List[str]]:
        return None

    def parse_metrics(self, stdout: str, stderr: str, artifact_dir: Path) -> dict:
        return {}
