import os
from pathlib import Path
import subprocess
import pytest


def test_completion_dependency_and_shutdown_admission(tmp_path):
    root = Path(__file__).resolve().parents[1]
    rocm = Path(os.environ.get('ROCM_PATH', '/opt/rocm'))
    if not (rocm / 'include/hsa/hsa.h').exists():
        pytest.skip('ROCm headers required for HSA worker unit test')
    binary = tmp_path / 'completion-test'
    subprocess.run(['g++', '-std=c++17', '-I', str(root / 'src'), '-I', str(rocm / 'include'),
                    '-DAMD_INTERNAL_BUILD', '-D__HIP_PLATFORM_AMD__',
                    str(root / 'tests/native_completion.cpp'), str(root / 'src/trace_db.cpp'),
                    str(root / 'src/roctx_shim.cpp'), str(root / 'src/hip_api_intercept.cpp'),
                    '-L'+str(rocm / 'lib'), '-Wl,-rpath,'+str(rocm / 'lib'),
                    '-lhsa-runtime64', '-lsqlite3', '-ldl', '-pthread', '-o', str(binary)],
                   check=True, capture_output=True, timeout=90)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
