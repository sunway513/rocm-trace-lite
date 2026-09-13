import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import pytest


def test_real_roctx_thread_and_nesting_contract(tmp_path):
    if not shutil.which('g++'):
        pytest.skip('g++ is required')
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / 'roctx-test'
    subprocess.run(['g++', '-std=c++17', '-I', str(root / 'src'),
                    str(root / 'tests/native_roctx.cpp'), str(root / 'src/trace_db.cpp'),
                    str(root / 'src/roctx_shim.cpp'), '-lsqlite3', '-pthread', '-o', str(binary)],
                   check=True, capture_output=True, timeout=60)
    path = tmp_path / 'trace.db'
    subprocess.run([str(binary)], env={**os.environ, 'RTL_OUTPUT': str(path)},
                   check=True, capture_output=True, timeout=30)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM rocpd_op').fetchone()[0] == 103
        assert db.execute('SELECT COUNT(DISTINCT roctxId) FROM rocpd_op').fetchone()[0] == 103
