"""Tests for HIP Graph safety (issue #15, ADR-001).

Validates timeout-based signal wait and clean shutdown.
CPU tests verify architecture in source code.
GPU tests verify no crash with graph replay.
"""
import os
import re
import sqlite3
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HSA_FILE = os.path.join(REPO_ROOT, "src", "hsa_intercept.cpp")
LIB_PATH = os.path.join(REPO_ROOT, "librtl.so")


def _has_gpu():
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch; assert torch.cuda.is_available()"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        return r.returncode == 0
    except Exception:
        return False


def _skip_if_no_gpu():
    if not _has_gpu() or not os.path.exists(LIB_PATH):
        import pytest
        pytest.skip("No GPU or librtl.so not built")


def _run_traced(script, trace_path, timeout=60):
    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = LIB_PATH
    env["RTL_OUTPUT"] = trace_path
    r = subprocess.run(
        [sys.executable, "-c", script],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout
    )
    return r.returncode, r.stderr.decode()


# ---- CPU tests: architecture validation ----

class TestTimeoutBasedWait:
    """Verify signal wait uses bounded timeout, not UINT64_MAX."""

    def _get_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def test_no_uint64_max_in_signal_wait(self):
        """completion_worker must not use UINT64_MAX timeout."""
        src = self._get_source()
        match = re.search(r'static void completion_worker\(\).*?\n\}', src, re.DOTALL)
        assert match, "Could not find completion_worker"
        body = match.group()
        assert "UINT64_MAX" not in body, \
            "completion_worker still uses UINT64_MAX — must use bounded timeout"

    def test_timeout_constant_defined(self):
        """A timeout constant (e.g., WAIT_TIMEOUT_NS) should be defined."""
        src = self._get_source()
        assert "WAIT_TIMEOUT" in src, "No timeout constant found"

    def test_shutdown_check_in_wait_loop(self):
        """Wait loop must check g_shutdown between timeouts."""
        src = self._get_source()
        match = re.search(r'static void completion_worker\(\).*?\n\}', src, re.DOTALL)
        body = match.group()
        assert "g_shutdown" in body, "No shutdown check in completion_worker wait loop"
        # Should be a while loop with timeout + shutdown check
        assert "while" in body, "No while loop for timeout-based wait"

    def test_abandoned_dispatch_handled(self):
        """Dispatches abandoned during shutdown must be deleted, not leaked."""
        src = self._get_source()
        match = re.search(r'static void completion_worker\(\).*?\n\}', src, re.DOTALL)
        body = match.group()
        assert "delete dd" in body, "No delete for abandoned dispatch"
        assert "continue" in body, "No continue after abandoning dispatch"


class TestShutdownSafety:
    """Verify shutdown prevents double-call and drains work queue."""

    def _get_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def test_double_shutdown_prevention(self):
        """shutdown() must guard against being called twice."""
        src = self._get_source()
        match = re.search(r'static void shutdown\(\).*?\n\}', src, re.DOTALL)
        assert match
        body = match.group()
        assert "shutdown_done" in body or "once_flag" in body, \
            "No double-shutdown prevention"

    def test_work_queue_drained(self):
        """shutdown() must drain pending items from work queue."""
        src = self._get_source()
        match = re.search(r'static void shutdown\(\).*?\n\}', src, re.DOTALL)
        body = match.group()
        assert "g_work_queue" in body, "shutdown does not touch work queue"
        assert "delete" in body, "shutdown does not delete pending items"

    def test_join_before_close(self):
        """Worker join must happen before DB close."""
        src = self._get_source()
        match = re.search(r'static void shutdown\(\).*?\n\}', src, re.DOTALL)
        body = match.group()
        join_pos = body.find("join()")
        close_pos = body.find("close()")
        assert join_pos < close_pos, "DB close before worker join"

    def test_adr_document_exists(self):
        """ADR-001 document must exist."""
        adr = os.path.join(REPO_ROOT, "docs", "ADR-001-hipgraph-safety.md")
        assert os.path.exists(adr), "Missing ADR-001 document"


# ---- GPU tests ----

class TestGraphNoCrash:
    """HIP Graph must not crash with rpd_lite loaded."""

    def test_graph_20_replays(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.db")
        script = """
import torch
x = torch.ones(64, 64, dtype=torch.int32, device="cuda")
s = torch.cuda.Stream()
with torch.cuda.stream(s): _ = x + 1
s.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s): y = x + 1
for _ in range(20): g.replay()
s.synchronize()
print("ok")
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Crashed with 20 replays: {stderr[-500:]}"

    def test_graph_100_gemm_replays(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.db")
        script = """
import torch
x = torch.randn(256, 256, device="cuda")
s = torch.cuda.Stream()
with torch.cuda.stream(s): _ = x @ x
s.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s): y = x @ x
for _ in range(100): g.replay()
s.synchronize()
print("ok")
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Crashed with 100 GEMM replays: {stderr[-500:]}"

    def test_graph_correctness(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.db")
        script = """
import torch
x = torch.ones(64, 64, dtype=torch.int32, device="cuda")
s = torch.cuda.Stream()
with torch.cuda.stream(s): _ = x + 1
s.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s): y = x + 1
g.replay(); s.synchronize()
assert y[0,0].item() == 2, f"Wrong value: {y[0,0].item()}"
print("ok")
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Correctness failed: {stderr[-500:]}"

    def test_mixed_eager_and_graph(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.db")
        script = """
import torch
x = torch.randn(128, 128, device="cuda")
for _ in range(10): y = x @ x
torch.cuda.synchronize()
s = torch.cuda.Stream()
with torch.cuda.stream(s): _ = x @ x
s.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s): y = x @ x
for _ in range(50): g.replay()
s.synchronize()
for _ in range(10): y = x @ x
torch.cuda.synchronize()
print("ok")
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Mixed workload crashed: {stderr[-500:]}"
        conn = sqlite3.connect(trace)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert count >= 10, f"Expected >=10 profiled ops, got {count}"
