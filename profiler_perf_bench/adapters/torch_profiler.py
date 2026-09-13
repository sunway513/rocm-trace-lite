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
    execution_model = ExecutionModel.EXTERNAL_WRAPPER

    def __init__(self):
        self._prof: Optional[Any] = None
        self._tmpdir: Optional[Path] = None

    def prepare_run(self, cmd: list, env: dict, tmpdir: Path) -> tuple:
        # Run the Python entrypoint in the same interpreter as the profiler.
        if not cmd or not Path(cmd[0]).name.startswith('python'):
            raise ValueError('torch_profiler requires a Python workload; native binaries are unsupported')
        bootstrap = str(Path(__file__).with_name('_torch_child.py'))
        return [cmd[0], bootstrap, str(tmpdir / 'torch_profiler_trace'), *cmd[1:]], env

    def start(self, tmpdir: Path) -> None:
        try:
            import torch
            from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        except ImportError:
            raise RuntimeError("torch is required for TorchProfilerAdapter")

        self._tmpdir = tmpdir
        trace_dir = str(tmpdir / "torch_profiler_trace")

        self._prof = profile(
            activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
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
        return hashlib.md5(b"torch_profiler:child:cpu+available_gpu:v2").hexdigest()
