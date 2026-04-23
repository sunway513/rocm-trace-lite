"""Shared base for L1 HIP workloads that use the gpu_workload binary."""

import os
from pathlib import Path
from typing import Optional, Callable, List
from ..base import Level, Workload

# Default path to the pre-compiled gpu_workload binary
_DEFAULT_BINARY = str(
    Path(__file__).parent.parent.parent.parent / "tests" / "gpu_workload"
)


class HipWorkloadBase(Workload):
    """Base class for L1 HIP workloads."""

    level = Level.L1

    def __init__(self, binary_path: Optional[str] = None):
        self._binary = binary_path or _DEFAULT_BINARY

    def env(self) -> dict:
        return {}

    def ready_probe(self) -> Optional[Callable[[], bool]]:
        return None

    def client_cmd(self) -> Optional[List[str]]:
        return None

    def parse_metrics(self, stdout: str, stderr: str, artifact_dir: Path) -> dict:
        """Minimal parse — wall time is measured by BenchmarkRunner."""
        return {}

    @property
    def requires(self) -> List[str]:
        return [self._binary]
