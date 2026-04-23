"""RTL adapter — HSA_TOOLS_LIB / LD_PRELOAD based profiler adapter.

Three variants registered:
  rtl          → RTL_MODE=lite   (no LD_PRELOAD)
  rtl_standard → RTL_MODE=standard (no LD_PRELOAD)
  rtl_hip      → RTL_MODE=hip + LD_PRELOAD

All three use the same underlying RTLAdapter class parameterized by mode.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

from .base import ExecutionModel, ProfilerAdapter
from .registry import global_registry


def _get_librtl_path() -> Optional[str]:
    """Find librtl.so path via rocm_trace_lite.get_lib_path() or env override."""
    # Allow override via env var for CI / container scenarios
    env_override = os.environ.get("RTL_LIB_PATH")
    if env_override and os.path.isfile(env_override):
        return env_override

    try:
        import rocm_trace_lite
        path = rocm_trace_lite.get_lib_path()
        if os.path.isfile(path):
            return path
    except Exception:
        pass

    # Fallback: look next to the main repo root
    repo_root = Path(__file__).parent.parent.parent
    candidates = [
        repo_root / "librtl.so",
        repo_root / "rocm_trace_lite" / "lib" / "librtl.so",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    return None


class RTLAdapter(ProfilerAdapter):
    """RTL profiler adapter.

    Injects HSA_TOOLS_LIB + RTL_MODE into the workload's env.
    For hip mode, also injects LD_PRELOAD for HIP Runtime API interception.
    """

    execution_model = ExecutionModel.EXTERNAL_WRAPPER

    def __init__(self, mode: str = "lite"):
        if mode not in ("lite", "standard", "hip"):
            raise ValueError(f"Unknown RTL mode: {mode!r}. Must be one of: lite, standard, hip")
        self._mode = mode
        self.name = {
            "lite": "rtl",
            "standard": "rtl_standard",
            "hip": "rtl_hip",
        }[mode]

    def prepare_run(self, cmd: list, env: dict, tmpdir: Path) -> tuple:
        lib_path = _get_librtl_path()
        if lib_path is None:
            raise RuntimeError(
                "librtl.so not found. Set RTL_LIB_PATH or ensure rocm_trace_lite is installed."
            )

        new_env = dict(env)
        new_env["HSA_TOOLS_LIB"] = lib_path
        new_env["RTL_MODE"] = self._mode
        new_env["RTL_OUTPUT"] = str(tmpdir / "trace.db")

        if self._mode == "hip":
            # HIP mode also needs LD_PRELOAD for HIP Runtime API interception
            existing_preload = new_env.get("LD_PRELOAD", "")
            if existing_preload:
                new_env["LD_PRELOAD"] = f"{lib_path}:{existing_preload}"
            else:
                new_env["LD_PRELOAD"] = lib_path

        return cmd, new_env

    def start(self, tmpdir: Path) -> None:
        pass  # external_wrapper: not used in-process

    def stop(self) -> None:
        pass  # external_wrapper: not used in-process

    def artifact_glob(self) -> str:
        return "trace.db"

    def config_hash(self) -> str:
        key = f"rtl:{self._mode}".encode()
        return hashlib.md5(key).hexdigest()


# Register the three RTL variants
@global_registry.register
class _RTLLiteAdapter(RTLAdapter):
    name = "rtl"

    def __init__(self):
        super().__init__(mode="lite")


@global_registry.register
class _RTLStandardAdapter(RTLAdapter):
    name = "rtl_standard"

    def __init__(self):
        super().__init__(mode="standard")


@global_registry.register
class _RTLHipAdapter(RTLAdapter):
    name = "rtl_hip"

    def __init__(self):
        super().__init__(mode="hip")
