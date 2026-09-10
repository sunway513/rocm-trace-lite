import os
from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.mark.parametrize('case', ['failure', 'skip', 'pass'])
def test_gpu_gate_rejects_failures_and_all_skips(tmp_path, case):
    if not shutil.which('bash'):
        pytest.skip('bash required')
    test = tmp_path / 'test_injected.py'
    if case == 'skip':
        test.write_text('import pytest\n@pytest.mark.skip\ndef test_skip(): pass\n')
    else:
        test.write_text('\n'.join(f'def test_pass_{i}(): pass' for i in range(6)) +
                        ('\ndef test_failure(): assert False\n' if case == 'failure' else '\n'))
    script = Path(__file__).resolve().parents[1] / 'tools/run_gpu_tests.sh'
    result = subprocess.run(['bash', str(script), str(tmp_path / 'result.log'), '5', str(test)],
                            env={**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'},
                            capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == (case == 'pass'), result.stdout + result.stderr
