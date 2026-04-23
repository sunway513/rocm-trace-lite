"""Base abstractions for workloads."""

import abc
import shutil
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional


class Level(Enum):
    """Workload complexity level."""
    L1 = 1  # HIP-native C++ binary, <60s, no Python deps
    L2 = 2  # Python (torch) workload, 1-3 min, requires torch+ROCm
    L3 = 3  # N2N LLM serving, 10-30 min, requires model weights


class Workload(abc.ABC):
    """Abstract base for all benchmark workloads."""

    name: str
    level: Level
    requires: List[str]

    @abc.abstractmethod
    def cmd(self) -> List[str]:
        """Return the command to run. For L3, this is the server command."""
        ...

    @abc.abstractmethod
    def env(self) -> dict:
        """Return environment variables for the workload process."""
        ...

    @abc.abstractmethod
    def ready_probe(self) -> Optional[Callable[[], bool]]:
        """Return a callable that returns True when server is ready, or None."""
        ...

    @abc.abstractmethod
    def client_cmd(self) -> Optional[List[str]]:
        """Return the client command for L3 workloads, or None for L1/L2."""
        ...

    @abc.abstractmethod
    def parse_metrics(self, stdout: str, stderr: str, artifact_dir: Path) -> dict:
        """Extract perf metrics from process output."""
        ...

    def check_requires(self) -> List[str]:
        """Return list of missing requirements from self.requires.

        Checks each item:
          - If it starts with '/', check it as a file path.
          - Otherwise, check it as a command in PATH via shutil.which().
        """
        missing = []
        for req in self.requires:
            if req.startswith("/"):
                if not Path(req).exists():
                    missing.append(req)
            else:
                if shutil.which(req) is None:
                    missing.append(req)
        return missing
