"""Real HIP records must survive the multiprocess merge (no torch required)."""
import os
from pathlib import Path
import sqlite3
import subprocess
import pytest
from rocm_trace_lite.cmd_trace import _merge_traces

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_process_records_survive_merge(tmp_path):
    binary = ROOT / 'tests/trace_regions'
    lib = ROOT / 'librtl.so'
    if not binary.exists() or not lib.exists():
        pytest.skip('Build tests/trace_regions and librtl.so')
    env = {**os.environ, 'HSA_TOOLS_LIB': str(lib), 'LD_PRELOAD': str(lib),
           'RTL_MODE': 'hip', 'RTL_OUTPUT': str(tmp_path / 'part_%p.db')}
    for _ in range(2):
        result = subprocess.run([str(binary)], env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    files = sorted(tmp_path.glob('part_*.db'))
    assert len(files) == 2

    def stats(path):
        with sqlite3.connect(path) as db:
            return (db.execute('SELECT COUNT(*) FROM rocpd_api').fetchone()[0],
                    db.execute('SELECT COUNT(*) FROM rocpd_op WHERE roctxId>0').fetchone()[0])

    before = [stats(p) for p in files]
    assert all(a > 0 and r >= 11 for a, r in before), before
    out = tmp_path / 'out.db'
    _merge_traces([str(p) for p in files], str(out))
    assert stats(out) == tuple(sum(row[i] for row in before) for i in range(2))
    with sqlite3.connect(out) as db:
        assert db.execute('SELECT COUNT(DISTINCT roctxId) FROM rocpd_op WHERE gpuId<0').fetchone()[0] == 4
        assert db.execute("SELECT COUNT(*) FROM rocpd_op o JOIN rocpd_string s ON s.id=o.description_id WHERE s.string='cross-thread-region'").fetchone()[0] == 2
