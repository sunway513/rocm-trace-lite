"""GPU E2E tests for roctx manual markers via rpd_lite shim (issue #23).

Validates that roctxRangePushA, roctxRangePop, and roctxMarkA symbols
exported by librpd_lite.so produce correct UserMarker records in the trace.

These tests require a ROCm GPU and librpd_lite.so built.
Run with: pytest tests/test_roctx_e2e.py -v
"""
import os
import sqlite3
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "librpd_lite.so")


def _has_gpu():
    """Check if ROCm GPU is available."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch; assert torch.cuda.is_available()"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        return r.returncode == 0
    except Exception:
        return False


def _has_lib():
    return os.path.exists(LIB_PATH)


def _skip_if_no_gpu():
    if not _has_gpu() or not _has_lib():
        import pytest
        pytest.skip("No GPU or librpd_lite.so not built")


def _run_traced(script, trace_path, timeout=120):
    """Run a Python script with rpd_lite tracing, return (returncode, stderr)."""
    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = LIB_PATH
    env["RPD_LITE_OUTPUT"] = trace_path
    r = subprocess.run(
        [sys.executable, "-c", script],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout
    )
    return r.returncode, r.stderr.decode()


def _get_user_markers(trace_path):
    """Query UserMarker records from an rpd trace file."""
    conn = sqlite3.connect(trace_path)
    markers = conn.execute(
        "SELECT s.string FROM rocpd_op o "
        "JOIN rocpd_string s ON o.description_id = s.id "
        "JOIN rocpd_string ot ON o.opType_id = ot.id "
        "WHERE ot.string = 'UserMarker'"
    ).fetchall()
    conn.close()
    return [m[0] for m in markers]


def _get_user_markers_with_duration(trace_path):
    """Query UserMarker records with duration from an rpd trace file."""
    conn = sqlite3.connect(trace_path)
    rows = conn.execute(
        "SELECT s.string, (o.end - o.start) as dur FROM rocpd_op o "
        "JOIN rocpd_string s ON o.description_id = s.id "
        "JOIN rocpd_string ot ON o.opType_id = ot.id "
        "WHERE ot.string = 'UserMarker'"
    ).fetchall()
    conn.close()
    return rows


# -- roctx shim loader snippet used in all subprocess scripts --
_ROCTX_LOADER = """
import ctypes, os
lib_path = os.environ.get("HSA_TOOLS_LIB", "librpd_lite.so")
roctx = ctypes.CDLL(lib_path)
roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
roctx.roctxRangePushA.restype = ctypes.c_int
roctx.roctxRangePop.restype = ctypes.c_int
roctx.roctxMarkA.argtypes = [ctypes.c_char_p]
"""


class TestRoctxPushPop:
    """roctxRangePushA / roctxRangePop E2E."""

    def test_roctx_push_pop_captured(self, tmp_path):
        """Push/Pop range around a matmul produces a UserMarker in the trace."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.rpd")
        script = _ROCTX_LOADER + """
import torch
roctx.roctxRangePushA(b"matmul_region")
x = torch.randn(256, 256, device="cuda")
y = x @ x
torch.cuda.synchronize()
roctx.roctxRangePop()
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Script failed: {stderr[-500:]}"
        assert os.path.exists(trace), "No trace file created"
        markers = _get_user_markers(trace)
        assert "matmul_region" in markers, (
            f"Expected 'matmul_region' in UserMarker records, got: {markers}"
        )


class TestRoctxMark:
    """roctxMarkA E2E."""

    def test_roctx_mark_captured(self, tmp_path):
        """roctxMarkA produces a zero-duration UserMarker."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.rpd")
        script = _ROCTX_LOADER + """
import torch
x = torch.randn(64, 64, device="cuda")
_ = x + 1
torch.cuda.synchronize()
roctx.roctxMarkA(b"checkpoint")
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Script failed: {stderr[-500:]}"
        rows = _get_user_markers_with_duration(trace)
        checkpoint_rows = [r for r in rows if r[0] == "checkpoint"]
        assert len(checkpoint_rows) >= 1, (
            f"Expected at least one 'checkpoint' marker, got: {rows}"
        )
        # Mark should have zero duration
        for msg, dur in checkpoint_rows:
            assert dur == 0, f"Mark '{msg}' should have duration=0, got {dur}"


class TestRoctxNested:
    """Nested roctx ranges."""

    def test_roctx_nested_ranges(self, tmp_path):
        """Push outer, push inner, matmul, pop, pop — both messages captured."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.rpd")
        script = _ROCTX_LOADER + """
import torch
roctx.roctxRangePushA(b"outer")
roctx.roctxRangePushA(b"inner")
x = torch.randn(256, 256, device="cuda")
y = x @ x
torch.cuda.synchronize()
roctx.roctxRangePop()
roctx.roctxRangePop()
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Script failed: {stderr[-500:]}"
        markers = _get_user_markers(trace)
        assert "outer" in markers, f"Missing 'outer' marker, got: {markers}"
        assert "inner" in markers, f"Missing 'inner' marker, got: {markers}"


class TestRoctxKernelCoexistence:
    """roctx markers + kernel dispatches in the same trace."""

    def test_roctx_with_kernel_coexistence(self, tmp_path):
        """Both UserMarker and kernel ops should appear in the trace."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.rpd")
        script = _ROCTX_LOADER + """
import torch
roctx.roctxRangePushA(b"compute_block")
x = torch.randn(256, 256, device="cuda")
for _ in range(10):
    x = x @ x
torch.cuda.synchronize()
roctx.roctxRangePop()
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Script failed: {stderr[-500:]}"

        # Check UserMarker present
        markers = _get_user_markers(trace)
        assert "compute_block" in markers, (
            f"Missing 'compute_block' marker, got: {markers}"
        )

        # Check kernel ops also present
        conn = sqlite3.connect(trace)
        total_ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        marker_count = conn.execute(
            "SELECT count(*) FROM rocpd_op o "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'UserMarker'"
        ).fetchone()[0]
        conn.close()
        kernel_ops = total_ops - marker_count
        assert kernel_ops >= 10, (
            f"Expected >=10 kernel ops alongside markers, got {kernel_ops} "
            f"(total={total_ops}, markers={marker_count})"
        )


class TestRoctxMultithread:
    """Multi-threaded roctx marker capture."""

    def test_roctx_multithread(self, tmp_path):
        """4 threads each push a named range + matmul + pop.
        All 4 distinct marker messages must appear in the trace."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.rpd")
        script = _ROCTX_LOADER + """
import torch, threading

def worker(name):
    roctx.roctxRangePushA(name.encode())
    x = torch.randn(128, 128, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    roctx.roctxRangePop()

names = ["thread_0", "thread_1", "thread_2", "thread_3"]
threads = [threading.Thread(target=worker, args=(n,)) for n in names]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize()
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Script failed: {stderr[-500:]}"
        markers = _get_user_markers(trace)
        for name in ["thread_0", "thread_1", "thread_2", "thread_3"]:
            assert name in markers, (
                f"Missing '{name}' marker in multi-thread trace, got: {markers}"
            )
