"""Exercise the CI shell entry point with controlled dynamic-loader results."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/check_native_dependencies.sh'
pytestmark = pytest.mark.skipif(shutil.which('bash') is None, reason='requires bash')


@pytest.mark.parametrize('output,status,expected', [
    ('libc.so.6 => /lib/libc.so.6 (0x1000)', 0, 0),
    ('libhsa-runtime64.so.1 => not found', 0, 1),
    ('libroctracer64.so.4 => /opt/rocm/lib/libroctracer64.so.4', 0, 1),
    ('librocprofiler-sdk.so => /opt/rocm/lib/librocprofiler-sdk.so', 0, 1),
    ('libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7', 0, 1),
    ('libroctx64.so.4 => /opt/rocm/lib/libroctx64.so.4', 0, 1),
    ('', 2, 2),
])
def test_dependency_gate(tmp_path, output, status, expected):
    # A fake loader lets the same workflow command cover failures without
    # installing forbidden libraries or damaging an actual native binary.
    loader = tmp_path / 'ldd'
    loader.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_LDD_OUTPUT"\nexit "$TEST_LDD_STATUS"\n')
    loader.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
               TEST_LDD_OUTPUT=output, TEST_LDD_STATUS=str(status))
    result = subprocess.run(['bash', str(SCRIPT), str(tmp_path / 'library with spaces.so')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stdout + result.stderr
