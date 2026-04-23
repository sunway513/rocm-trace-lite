"""L2 inference simulation torch workload.

Per spec §4.2: L2-inference-sim, uses benchmarks/workloads/inference_sim.py
"""

import sys
from pathlib import Path
from typing import Optional, Callable, List

from ..base import Level, Workload

_INFERENCE_SIM_SCRIPT = str(
    Path(__file__).parent.parent.parent.parent / "benchmarks" / "workloads" / "inference_sim.py"
)


class InferenceSimTorchWorkload(Workload):
    """L2-inference-sim: torch-based inference simulation workload."""

    name = "L2-inference-sim"
    level = Level.L2
    requires = ["torch", _INFERENCE_SIM_SCRIPT]

    def cmd(self) -> List[str]:
        return [sys.executable, _INFERENCE_SIM_SCRIPT]

    def env(self) -> dict:
        return {}

    def ready_probe(self) -> Optional[Callable[[], bool]]:
        return None

    def client_cmd(self) -> Optional[List[str]]:
        return None

    def parse_metrics(self, stdout: str, stderr: str, artifact_dir: Path) -> dict:
        return {}
