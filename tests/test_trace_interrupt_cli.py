"""CPU-only CLI integration: real process groups and real SQLite trace merging."""
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import pytest
from conftest import SCHEMA_SQL


BOOTSTRAP = '''import rocm_trace_lite
from rocm_trace_lite import cmd_trace
# Replace only the native dependency; parser, runner, merge and export are real.
rocm_trace_lite.get_lib_path = lambda: ""
cmd_trace._preflight_check = lambda _: None
from rocm_trace_lite.cli import main
main()
'''
WORKLOAD = '''import os, signal, sqlite3, subprocess, sys, time
from pathlib import Path
worker = len(sys.argv) > 1 and sys.argv[1] == "worker"
if worker:
    signal.signal(signal.SIGINT, lambda *_: Path("interrupted").touch())
db = sqlite3.connect(os.environ["RTL_OUTPUT"].replace("%p", str(os.getpid())))
db.executescript(Path("schema.sql").read_text())
db.executemany("INSERT INTO rocpd_string VALUES(?,?)", [(1,"kernel"),(2,"KernelExecution")])
db.commit()
if worker:
    Path("ready").write_text(str(os.getpid()))
    time.sleep(20 if os.environ.get("SECOND_INTERRUPT") else 0.8)
else:
    subprocess.Popen([sys.executable, __file__, "worker"])
    Path("stdin.txt").write_text(sys.stdin.readline())
db.execute("INSERT INTO rocpd_op(id,gpuId,queueId,start,end,description_id,opType_id) VALUES(1,0,1,1,2,1,2)")
db.execute("INSERT INTO rocpd_metadata(tag,value) VALUES('pid',?)", (str(os.getpid()),))
db.commit()
db.close()
raise SystemExit(0 if worker else int(os.environ.get("WORKLOAD_EXIT", "0")))
'''


def wait_file(path, proc):
    deadline = time.monotonic() + 10
    while not path.exists():
        assert proc.poll() is None
        assert time.monotonic() < deadline
        time.sleep(0.01)


def start_cli(tmp_path, code=0, second=False):
    (tmp_path / "schema.sql").write_text(SCHEMA_SQL)
    (tmp_path / "workload.py").write_text(WORKLOAD)
    (tmp_path / "cli.py").write_text(BOOTSTRAP)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
           "WORKLOAD_EXIT": str(code)}
    env.pop("LD_PRELOAD", None)
    if second:
        env["SECOND_INTERRUPT"] = "1"
    log = open(tmp_path / "cli.log", "w")
    proc = subprocess.Popen([sys.executable, "cli.py", "trace", "-o", "merged.db",
                             "--", sys.executable, "workload.py"],
                            cwd=tmp_path, env=env, stdin=subprocess.PIPE, stdout=log,
                            stderr=log, text=True)
    proc.stdin.write("stdin survives a new session\n")
    proc.stdin.close()
    return proc, log


def cleanup(proc, log, tmp_path):
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    ready = tmp_path / "ready"
    if ready.exists():
        try:
            os.kill(int(ready.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass
    log.close()


@pytest.mark.parametrize("code,interrupt", [(0, False), (7, False), (0, True)])
def test_cli_collects_delayed_sqlite_worker(tmp_path, code, interrupt):
    proc, log = start_cli(tmp_path, code)
    try:
        wait_file(tmp_path / "ready", proc)
        if interrupt:
            proc.send_signal(signal.SIGINT)
            wait_file(tmp_path / "interrupted", proc)
        assert proc.wait(timeout=10) == (130 if interrupt else code), (tmp_path / "cli.log").read_text()
        assert (tmp_path / "stdin.txt").read_text() == "stdin survives a new session\n"
        with sqlite3.connect(tmp_path / "merged.db") as db:
            assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert not db.execute("PRAGMA foreign_key_check").fetchall()
            assert db.execute("SELECT COUNT(*) FROM rocpd_op").fetchone()[0] == 2
        assert (tmp_path / "merged.json.gz").exists()
        assert not list(tmp_path.glob("merged_[0-9]*.db"))
    finally:
        cleanup(proc, log, tmp_path)


def test_second_interrupt_preserves_sqlite_inputs(tmp_path):
    proc, log = start_cli(tmp_path, second=True)
    try:
        wait_file(tmp_path / "ready", proc)
        proc.send_signal(signal.SIGINT)
        wait_file(tmp_path / "interrupted", proc)
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) != 0
        assert not (tmp_path / "merged.db").exists()
        raw = list(tmp_path.glob("merged_[0-9]*.db"))
        assert len(raw) == 2
        for path in raw:
            with sqlite3.connect(path) as db:
                assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert "raw trace files retained" in (tmp_path / "cli.log").read_text()
    finally:
        cleanup(proc, log, tmp_path)


def test_terminal_stdin_and_ctrl_c_forwarding(tmp_path):
    import pty

    (tmp_path / "schema.sql").write_text(SCHEMA_SQL)
    (tmp_path / "workload.py").write_text(WORKLOAD)
    (tmp_path / "cli.py").write_text(BOOTSTRAP)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    env.pop("LD_PRELOAD", None)
    pid, terminal = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execvpe(sys.executable, [sys.executable, "cli.py", "trace", "-o", "merged.db",
                                  "--", sys.executable, "workload.py"], env)
    reaped = False
    try:
        os.write(terminal, b"stdin survives a new session\n")
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists() or not (tmp_path / "stdin.txt").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # Terminal line discipline delivers SIGINT to the CLI foreground group.
        os.write(terminal, b"\x03")
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                reaped = True
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 130
        assert (tmp_path / "interrupted").exists()
        assert (tmp_path / "stdin.txt").read_text() == "stdin survives a new session\n"
        with sqlite3.connect(tmp_path / "merged.db") as db:
            assert db.execute("SELECT COUNT(*) FROM rocpd_op").fetchone()[0] == 2
    finally:
        os.close(terminal)
        if not reaped:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        if (tmp_path / "ready").exists():
            try:
                os.kill(int((tmp_path / "ready").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
