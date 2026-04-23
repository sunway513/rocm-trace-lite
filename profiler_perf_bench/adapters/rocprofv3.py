"""rocprofv3 adapter — rocprofiler-sdk command prefix."""

import hashlib
from pathlib import Path
from .base import ExecutionModel, ProfilerAdapter
from .registry import global_registry


@global_registry.register
class RocprofV3Adapter(ProfilerAdapter):
    """rocprofv3 external-wrapper adapter.

    Prepends: rocprofv3 --runtime-trace -o <out> -- <original_cmd>
    """

    name = "rocprofv3"
    execution_model = ExecutionModel.EXTERNAL_WRAPPER

    def prepare_run(self, cmd: list, env: dict, tmpdir: Path) -> tuple:
        out_dir = str(tmpdir / "rocprofv3_out")
        new_cmd = [
            "rocprofv3",
            "--runtime-trace",
            "-o", out_dir,
            "--",
        ] + cmd
        return new_cmd, dict(env)

    def start(self, tmpdir: Path) -> None:
        pass

    def stop(self) -> None:
        pass

    def artifact_glob(self) -> str:
        return "rocprofv3_out*"

    def config_hash(self) -> str:
        return hashlib.md5(b"rocprofv3:runtime-trace").hexdigest()
