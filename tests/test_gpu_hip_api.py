"""GPU integration tests for HIP API interception.

Requires: ROCm GPU + librtl.so built with HIP API support.
Run with: RTL_MODE=hip python3 -m pytest tests/test_gpu_hip_api.py -v --timeout=120
"""
import os
import sqlite3
import subprocess
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "librtl.so")
WORKLOAD = os.path.join(REPO_ROOT, "tests", "gpu_workload")

HAS_LIB = os.path.exists(LIB_PATH)
HAS_WORKLOAD = os.path.exists(WORKLOAD)


def _run_with_rtl(mode, workload_args, extra_env=None):
    """Run gpu_workload under RTL with given mode, return trace DB path."""
    if not HAS_LIB or not HAS_WORKLOAD:
        pytest.skip("librtl.so or gpu_workload not built")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = LIB_PATH
    env["LD_PRELOAD"] = LIB_PATH
    env["RTL_OUTPUT"] = db_path
    env["RTL_MODE"] = mode
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [WORKLOAD] + workload_args,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    return db_path, result


@pytest.mark.skipif(not HAS_LIB or not HAS_WORKLOAD,
                    reason="Need librtl.so and gpu_workload")
class TestHipApiCapture:

    def test_hip_api_capture_basic(self):
        db, r = _run_with_rtl("hip", ["gemm", "256", "10"])
        assert r.returncode == 0, f"Workload failed: {r.stderr}"
        conn = sqlite3.connect(db)
        api_count = conn.execute("SELECT count(*) FROM rocpd_api").fetchone()[0]
        assert api_count > 0, "No HIP API events captured"
        api_names = [row[0] for row in conn.execute(
            "SELECT s.string FROM rocpd_api a "
            "JOIN rocpd_string s ON a.apiName_id = s.id").fetchall()]
        launch_apis = [n for n in api_names if "Launch" in n or "Module" in n]
        assert len(launch_apis) > 0, f"No launch APIs found. Got: {set(api_names)}"
        conn.close()
        os.unlink(db)

    def test_hip_api_capture_memcpy(self):
        db, r = _run_with_rtl("hip", ["memcpy", "1048576", "5"])
        assert r.returncode == 0
        conn = sqlite3.connect(db)
        api_names = [row[0] for row in conn.execute(
            "SELECT s.string FROM rocpd_api a "
            "JOIN rocpd_string s ON a.apiName_id = s.id").fetchall()]
        memcpy_apis = [n for n in api_names if "Memcpy" in n or "memcpy" in n]
        assert len(memcpy_apis) > 0, f"No memcpy APIs. Got: {set(api_names)}"
        conn.close()
        os.unlink(db)

    def test_hip_api_timing(self):
        db, r = _run_with_rtl("hip", ["gemm", "256", "5"])
        assert r.returncode == 0
        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT start, end FROM rocpd_api").fetchall()
        assert len(rows) > 0
        for start, end in rows:
            assert start > 0, "start must be positive"
            assert end >= start, f"end ({end}) < start ({start})"
        conn.close()
        os.unlink(db)

    def test_hip_api_pid_tid(self):
        db, r = _run_with_rtl("hip", ["gemm", "256", "3"])
        assert r.returncode == 0
        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT pid, tid FROM rocpd_api").fetchall()
        assert len(rows) > 0
        for pid, tid in rows:
            assert pid > 0, "pid must be positive"
            assert tid > 0, "tid must be positive"
        conn.close()
        os.unlink(db)

    def test_hip_api_disabled_by_default(self):
        db, r = _run_with_rtl("standard", ["gemm", "256", "5"])
        assert r.returncode == 0
        conn = sqlite3.connect(db)
        api_count = conn.execute("SELECT count(*) FROM rocpd_api").fetchone()[0]
        assert api_count == 0, f"HIP API should be empty in standard mode, got {api_count}"
        conn.close()
        os.unlink(db)

    def test_gpu_ops_still_captured_in_hip_mode(self):
        db, r = _run_with_rtl("hip", ["gemm", "256", "10"])
        assert r.returncode == 0
        conn = sqlite3.connect(db)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        assert op_count > 0, "GPU kernel ops must still be captured in hip mode"
        conn.close()
        os.unlink(db)
