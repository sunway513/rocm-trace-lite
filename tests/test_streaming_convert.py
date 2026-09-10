"""Packaged converter compatibility, atomic output and event-count scaling."""
import gzip
import json
from pathlib import Path
import sqlite3

import pytest
from conftest import SCHEMA_SQL
from rocm_trace_lite import cmd_convert, cmd_trace


CASES = ['single', 'multi', 'hwq', 'queues101', 'api_only', 'empty']


def create_trace(path, case):
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_SQL)
        db.executemany('INSERT INTO rocpd_string(id,string) VALUES(?,?)', [
            (1, 'kernel_' + 'long_' * 30 + '.kd'), (2, 'KernelExecution'),
            (3, 'hipLaunchKernel'), (4, 'π = 3'), (5, 'normal_雪'),
            (6, 'kernel_UserArgs_suffix.kd'),
        ])
        count = 101 if case == 'queues101' else 7
        if case not in ('api_only', 'empty'):
            for i in range(count):
                gpu = [None, -1, 0, 2][i % 4] if case == 'multi' else 0
                info = ('hwq=q%d wg=64,1,1 grid=2,1,1' % (i % 2)) if case == 'hwq' and i != 0 else None
                db.execute('INSERT INTO rocpd_op(gpuId,queueId,start,end,description_id,opType_id,completionSignal) VALUES(?,?,?,?,?,?,?)',
                           (gpu, i if case == 'queues101' else i % 2,
                            1000 + (count-i)*100, 1050 + (count-i)*100,
                            [1, 5, 6][i % 3], 2, info))
            # Invalid duration must stay omitted; missing string joins also omitted.
            db.execute('INSERT INTO rocpd_op(gpuId,start,end,description_id) VALUES(0,10,10,1)')
            db.execute('INSERT INTO rocpd_op(gpuId,start,end,description_id) VALUES(0,1000,1010,999)')
        if case != 'empty':
            for pid, start, end in [(5, 1040, 1090), (None, 1050, 1040), (5, None, 1060)]:
                db.execute('INSERT INTO rocpd_api(pid,tid,start,end,apiName_id,args_id) VALUES(?,?,?,?,3,4)',
                           (pid, None, start, end))


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('compressed', [False, True])
def test_matches_pre_streaming_packaged_converter(tmp_path, case, compressed):
    db = tmp_path / 'trace.db'
    create_trace(db, case)
    out = tmp_path / ('trace.json.gz' if compressed else 'trace.json')
    cmd_convert.convert(str(db), str(out))
    expected = json.loads((Path(__file__).parent / 'fixtures/streaming-convert-golden.json').read_text())[case]
    if expected is None:
        assert not out.exists()
    else:
        opener = gzip.open if compressed else open
        with opener(out, 'rt') as stream:
            actual = json.load(stream)
        # Host metadata came from a set in the previous converter, so its order
        # varies between processes (notably for a None PID). Keep every event
        # and assert exact ordering for all GPU metadata and timed events.
        def stable_host_metadata(trace):
            events = trace['traceEvents']
            hosts = [e for e in events if e['ph'] == 'M' and e['pid'] >= 1000000]
            return [e for e in events if e not in hosts] + sorted(hosts, key=lambda e: e['pid'])
        assert stable_host_metadata(actual) == stable_host_metadata(expected)


@pytest.mark.parametrize('compressed', [False, True])
def test_output_failure_preserves_previous_file_and_removes_temp(tmp_path, monkeypatch, compressed):
    db = tmp_path / 'trace.db'
    create_trace(db, 'single')
    out = tmp_path / ('trace.json.gz' if compressed else 'trace.json')
    out.write_bytes(b'previous complete output')
    before = set(tmp_path.iterdir())
    original = cmd_convert.json.dumps
    calls = 0

    def fail(event, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError('simulated write/serialization failure')
        return original(event, *args, **kwargs)

    monkeypatch.setattr(cmd_convert.json, 'dumps', fail)
    with pytest.raises(OSError, match='simulated'):
        cmd_convert.convert(str(db), str(out))
    assert out.read_bytes() == b'previous complete output'
    assert set(tmp_path.iterdir()) == before
    with sqlite3.connect(db) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone() == ('ok',)


def test_trace_helper_requests_gzip_directly(tmp_path, monkeypatch):
    db = tmp_path / 'trace.db'
    db.touch()
    out = tmp_path / 'trace.json.gz'
    calls = []
    monkeypatch.setattr(cmd_convert, 'convert', lambda source, target: calls.append((source, target)))
    cmd_trace._generate_perfetto(str(db), str(out))
    assert calls == [(str(db), str(out))]
    assert set(tmp_path.iterdir()) == {db}


def test_conversion_rss_does_not_scale_with_dispatch_count(tmp_path):
    import os
    import subprocess
    import sys

    peaks = []
    for count in (10000, 100000):
        db = tmp_path / f'scale-{count}.db'
        with sqlite3.connect(db) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute('INSERT INTO rocpd_string VALUES(1,?)', ('kernel_' + 'long_name_' * 60,))
            conn.execute('INSERT INTO rocpd_string VALUES(2,?)', ('KernelExecution',))
            conn.executemany('INSERT INTO rocpd_op(gpuId,queueId,start,end,description_id,opType_id) VALUES(0,?,?,?,1,2)',
                             ((i, i*100, i*100+50) for i in range(count)))
        code = '''import json,resource,sys
from rocm_trace_lite.cmd_convert import convert
resource.setrlimit(resource.RLIMIT_AS, (192*1024*1024, 192*1024*1024))
convert(sys.argv[1], sys.argv[2])
print(json.dumps({"maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
'''
        result = subprocess.run([sys.executable, '-c', code, str(db), str(tmp_path / f'scale-{count}.json.gz')],
                                env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])},
                                capture_output=True, text=True, timeout=60, check=True)
        assert f'GPU ops:   {count}' in result.stdout
        assert f'Total events: {2*count+1}' in result.stdout
        peaks.append(json.loads(result.stdout.splitlines()[-1])['maxrss_kib'])
    # Ten times the rows and distinct queue IDs may grow SQLite buffers, not
    # a Python tuple/dict per event. Leave room for platform allocator variance.
    print("RSS KiB for 10k/100k unique-queue dispatches:", peaks)
    assert peaks[1] < peaks[0] + 32*1024, peaks
