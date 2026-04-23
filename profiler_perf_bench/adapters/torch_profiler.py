"""torch.profiler in-process adapter."""

import hashlib
from pathlib import Path
from typing import Optional, Any

from .base import ExecutionModel, ProfilerAdapter
from .registry import global_registry


@global_registry.register
class TorchProfilerAdapter(ProfilerAdapter):
    """In-process torch.profiler adapter.

    Uses torch.profiler.profile() context manager.
    Start/stop are called by BenchmarkRunner around the workload's cmd (in-process).
    """

    name = "torch_profiler"
    execution_model = ExecutionModel.IN_PROCESS_PYTHON

    def __init__(self):
        self._prof: Optional[Any] = None
        self._tmpdir: Optional[Path] = None

    def prepare_run(self, cmd: list, env: dict, tmpdir: Path) -> tuple:
        # In-process adapter: no cmd/env modification
        return cmd, env

    def start(self, tmpdir: Path) -> None:
        try:
            import torch
            from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        except ImportError:
            raise RuntimeError("torch is required for TorchProfilerAdapter")

        self._tmpdir = tmpdir
        trace_dir = str(tmpdir / "torch_profiler_trace")

        self._prof = profile(
            activities=[ProfilerActivity.CPU],
            on_trace_ready=tensorboard_trace_handler(trace_dir),
            record_shapes=False,
            with_stack=False,
        )
        self._prof.__enter__()

    def stop(self) -> None:
        if self._prof is not None:
            self._prof.__exit__(None, None, None)
            self._prof = None

    def artifact_glob(self) -> str:
        return "torch_profiler_trace/**/*.json"

    def config_hash(self) -> str:
        return hashlib.md5(b"torch_profiler:cpu").hexdigest()
