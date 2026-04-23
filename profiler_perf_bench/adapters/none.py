"""NoneAdapter — baseline with no profiler attached."""

import hashlib
from pathlib import Path
from .base import ExecutionModel, ProfilerAdapter
from .registry import global_registry


@global_registry.register
class NoneAdapter(ProfilerAdapter):
    """No-op adapter: passes cmd and env through unchanged.

    Used as the baseline measurement for computing overhead.
    """

    name = "none"
    execution_model = ExecutionModel.EXTERNAL_WRAPPER

    def prepare_run(self, cmd: list, env: dict, tmpdir: Path) -> tuple:
        """Identity pass-through — no modification."""
        return cmd, env

    def start(self, tmpdir: Path) -> None:
        pass  # no-op

    def stop(self) -> None:
        pass  # no-op

    def artifact_glob(self) -> str:
        return ""  # no artifacts produced

    def config_hash(self) -> str:
        return hashlib.md5(b"none").hexdigest()
