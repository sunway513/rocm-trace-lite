import sqlite3
import pytest
from conftest import SCHEMA_SQL
from rocm_trace_lite.cmd_trace import _merge_traces


def trace(path, pid, api_only=False, legacy=False):
    db = sqlite3.connect(path)
    schema = SCHEMA_SQL.replace(',\n    roctxId INTEGER DEFAULT 0', '') if legacy else SCHEMA_SQL
    db.executescript(schema)
    db.executemany('INSERT INTO rocpd_string VALUES(?,?)',
                   [(1, 'kernel'), (2, 'KernelExecution'), (3, 'range'),
                    (4, 'UserMarker'), (5, 'hipLaunch'), (6, str(pid))])
    db.execute('INSERT INTO rocpd_api VALUES(1,?,1,1,2,5,6)', (pid,))
    if not api_only:
        for ident, gpu, desc, typ in [(1, 0, 1, 2), (2, -1, 3, 4)]:
            db.execute('INSERT INTO rocpd_op(id,gpuId,start,end,description_id,opType_id) VALUES(?,?,1,2,?,?)',
                       (ident, gpu, desc, typ))
        if not legacy:
            db.execute('UPDATE rocpd_op SET roctxId=1')
        db.execute('INSERT INTO rocpd_api_ops VALUES(1,1,1)')
        db.execute('INSERT INTO rocpd_kernelapi(api_id,kernelName_id) VALUES(1,1)')
        db.execute('INSERT INTO rocpd_copyapi(api_id,size) VALUES(1,123)')
    db.execute('INSERT INTO rocpd_metadata(tag,value) VALUES(?,?)', ('pid', str(pid)))
    db.commit()
    return db


@pytest.mark.parametrize('legacy', [False, True])
def test_full_merge_and_reference_remapping(tmp_path, legacy):
    inputs = [str(tmp_path / f'{i}.db') for i in range(3)]
    for i, path in enumerate(inputs):
        trace(path, 100+i, api_only=i == 2, legacy=legacy).close()
    out = str(tmp_path / 'merged.db')
    _merge_traces(inputs, out)
    with sqlite3.connect(out) as db:
        assert db.execute('SELECT pid FROM rocpd_api ORDER BY pid').fetchall() == [(100,), (101,), (102,)]
        assert db.execute('SELECT COUNT(*) FROM rocpd_api_ops').fetchone()[0] == 2
        assert db.execute('SELECT COUNT(*) FROM rocpd_kernelapi').fetchone()[0] == 2
        assert db.execute('SELECT COUNT(*) FROM rocpd_copyapi').fetchone()[0] == 2
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        if not legacy:
            assert db.execute('SELECT COUNT(DISTINCT roctxId) FROM rocpd_op').fetchone()[0] == 2
            assert db.execute('SELECT COUNT(*) FROM rocpd_op k JOIN rocpd_op m ON k.roctxId=m.roctxId WHERE k.gpuId=0 AND m.gpuId=-1').fetchone()[0] == 2
    assert all(__import__('pathlib').Path(p).exists() for p in inputs)


def test_committed_wal_included(tmp_path):
    path = str(tmp_path / 'wal.db')
    db = trace(path, 100)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('INSERT INTO rocpd_api(pid) VALUES(200)')
    db.commit()
    out = str(tmp_path / 'out.db')
    try:
        _merge_traces([path], out)
        with sqlite3.connect(out) as merged:
            assert merged.execute('SELECT COUNT(*) FROM rocpd_api').fetchone()[0] == 2
    finally:
        db.close()


def test_failure_preserves_inputs_and_previous_output(tmp_path):
    good, bad, out = [tmp_path / n for n in ('good.db', 'bad.db', 'out.db')]
    trace(str(good), 100).close()
    bad.write_bytes(b'not a sqlite database')
    out.write_bytes(b'previous output')
    with pytest.raises(sqlite3.DatabaseError):
        _merge_traces([str(good), str(bad)], str(out))
    assert good.exists() and bad.exists()
    assert out.read_bytes() == b'previous output'
