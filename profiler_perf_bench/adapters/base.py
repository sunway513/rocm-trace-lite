"""Base abstractions for profiler adapters.

ProfilerAdapter is the plug-in interface — new adapters subclass this,
implement either prepare_run() (external_wrapper) or start/stop (in_process_python),
decorate with @register_adapter, and that's it (~30 lines).
"""

import abc
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Optional


class ExecutionModel(Enum):
    """How the profiler is attached to the workload process."""
    EXTERNAL_WRAPPER = "external_wrapper"    # command prefix + env injection
    IN_PROCESS_PYTHON = "in_process_python"  # torch.profiler, kineto


class ProfilerAdapter(abc.ABC):
    """Abstract base for all profiler adapters.

    External-wrapper adapters implement prepare_run().
    In-process adapters implement start()/stop().
    All adapters implement artifact_glob() and config_hash().
    """

    name: str
    execution_model: ExecutionModel

    # ── External-wrapper contract ──────────────────────────────────────────

    @abc.abstractmethod
    def prepare_run(self, cmd: list, env: dict, tmpdir: Path) -> tuple:
        """Return (modified_cmd, modified_env). No subprocess launch."""
        ...

    # ── In-process-python contract ─────────────────────────────────────────

    @abc.abstractmethod
    def start(self, tmpdir: Path) -> None:
        """Start in-process profiling. Noop for external-wrapper adapters."""
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop in-process profiling. Noop for external-wrapper adapters."""
        ...

    # ── Common ─────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def artifact_glob(self) -> str:
        """Glob pattern for trace outputs under tmpdir. Empty string = no artifacts."""
        ...

    @abc.abstractmethod
    def config_hash(self) -> str:
        """Deterministic hash of adapter configuration for reporting."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.execution_model.value})"
