"""rocprof (legacy roctracer) adapter — command prefix."""

import hashlib
from pathlib import Path
from .base import ExecutionModel, ProfilerAdapter
from .registry import global_registry


@global_registry.register
class RocprofAdapter(ProfilerAdapter):
    """Legacy rocprof external-wrapper adapter.

    Prepends: rocprof --hip-trace -o <out.csv> <original_cmd>
    """

    name = "rocprof"
    execution_model = ExecutionModel.EXTERNAL_WRAPPER

    def prepare_run(self, cmd: list, env: dict, tmpdir: Path) -> tuple:
        out_csv = str(tmpdir / "rocprof_out.csv")
        new_cmd = [
            "rocprof",
            "--hip-trace",
            "-o", out_csv,
        ] + cmd
        return new_cmd, dict(env)

    def start(self, tmpdir: Path) -> None:
        pass

    def stop(self) -> None:
        pass

    def artifact_glob(self) -> str:
        return "rocprof_out*.csv"

    def config_hash(self) -> str:
        return hashlib.md5(b"rocprof:hip-trace").hexdigest()
