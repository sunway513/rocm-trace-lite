"""Real CPU processes exercise CLI teardown without loading HIP or RTL."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from rocm_trace_lite.cmd_trace import _run_workload


def wait_file(path, process):
    deadline = time.monotonic() + 10
    while not path.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, str(path)
        time.sleep(0.01)


@pytest.mark.parametrize("code", [0, 7])
def test_exit_code(code):
    assert _run_workload([sys.executable, "-c", f"raise SystemExit({code})"], os.environ.copy()) == code


def test_signal_exit_code():
    assert _run_workload([sys.executable, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGTERM)"], os.environ.copy()) == 143


@pytest.mark.parametrize("interrupt", [False, True])
def test_waits_for_orphan_worker_before_collection(tmp_path, interrupt):
    worker = tmp_path / "worker.py"
    worker.write_text('''import os, signal, time
from pathlib import Path
signal.signal(signal.SIGINT, lambda *_: None)
Path("ready").write_text(str(os.getpid()))
time.sleep(0.7)
Path("trace_worker.db").write_text("fully flushed")
''')
    parent = tmp_path / "parent.py"
    parent.write_text('''import subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, "worker.py"])
while not Path("ready").exists(): time.sleep(0.01)
''')
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text('''import os, sys
from pathlib import Path
from rocm_trace_lite.cmd_trace import _run_workload
code = _run_workload([sys.executable, "parent.py"], os.environ.copy())
# This models the collection boundary: the final worker data must exist first.
assert Path("trace_worker.db").read_text() == "fully flushed"
Path("collected").write_text(str(code))
raise SystemExit(code)
''')
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.Popen([sys.executable, str(wrapper)], cwd=tmp_path, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_file(tmp_path / "ready", proc)
        if interrupt:
            proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode == (130 if interrupt else 0), (stdout, stderr)
        assert (tmp_path / "collected").read_text() == str(proc.returncode)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if (tmp_path / "ready").exists():
            try:
                os.kill(int((tmp_path / "ready").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("exit_code", [0, 9, 130])
def test_cli_preserves_status_without_trace(monkeypatch, tmp_path, exit_code):
    from types import SimpleNamespace
    import rocm_trace_lite
    from rocm_trace_lite import cmd_trace

    monkeypatch.setattr(rocm_trace_lite, "get_lib_path", lambda: "")
    monkeypatch.setattr(cmd_trace, "_preflight_check", lambda _: None)
    monkeypatch.setattr(cmd_trace, "_run_workload", lambda *_: exit_code)
    args = SimpleNamespace(cmd=["unused"], output=str(tmp_path / "trace.db"))
    with pytest.raises(SystemExit) as error:
        cmd_trace.run_trace(args)
    assert error.value.code == exit_code


def test_second_interrupt_does_not_collect_live_database(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text('''import os, signal, time
from pathlib import Path
signal.signal(signal.SIGINT, lambda *_: Path("interrupted").touch())
Path("trace_123.db").write_text("partial")
Path("ready").write_text(str(os.getpid()))
time.sleep(20)
''')
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text('''import os, sys
from pathlib import Path
from rocm_trace_lite.cmd_trace import _run_workload
_run_workload([sys.executable, "worker.py"], os.environ.copy())
Path("collected").touch()
''')
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.Popen([sys.executable, str(wrapper)], cwd=tmp_path, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_file(tmp_path / "ready", proc)
        proc.send_signal(signal.SIGINT)
        wait_file(tmp_path / "interrupted", proc)
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        # The deliberately live worker inherited the wrapper pipe descriptors.
        os.kill(int((tmp_path / "ready").read_text()), signal.SIGKILL)
        _, stderr = proc.communicate(timeout=10)
        assert proc.returncode != 0
        assert "raw trace files retained" in stderr
        assert (tmp_path / "trace_123.db").read_text() == "partial"
        assert not (tmp_path / "collected").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if (tmp_path / "ready").exists():
            try:
                os.kill(int((tmp_path / "ready").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
