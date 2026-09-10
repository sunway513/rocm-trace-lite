import os
from pathlib import Path
import subprocess
import pytest


@pytest.mark.parametrize('repeat', range(3))
def test_shutdown_drains_real_gpu_work(tmp_path, repeat):
    root = Path(__file__).resolve().parents[1]
    binary, lib = root / 'tests/shutdown_workload', root / 'librtl.so'
    if not binary.exists() or not lib.exists():
        pytest.skip('Build librtl.so and tests/shutdown_workload')
    env = {**os.environ, 'HSA_TOOLS_LIB': str(lib), 'LD_PRELOAD': str(lib),
           'RTL_MODE': 'standard', 'RTL_OUTPUT': str(tmp_path / 'shutdown.db')}
    result = subprocess.run([str(binary)], env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'shutdown_returned done=1' in result.stdout
