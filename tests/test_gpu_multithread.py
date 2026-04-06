"""GPU integration tests: multi-thread and multi-stream safety.

These tests require a ROCm GPU. Skip gracefully if unavailable.
Run with: pytest tests/test_gpu_multithread.py -v
"""
import os
import sqlite3
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "librtl.so")


def _has_gpu():
    """Check if ROCm GPU is available via PyTorch subprocess."""
    from conftest import _rocm_gpu_available
    return _rocm_gpu_available()


def _has_lib():
    return os.path.exists(LIB_PATH)


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


def _skip_if_no_gpu():
    if not _has_gpu() or not _has_lib():
        import pytest
        pytest.skip("No GPU or librtl.so not built")


class TestMultiStreamSingleGPU:
    """Multi-stream dispatches on a single GPU."""

    def test_4_streams_no_crash(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.db")
        script = """
import torch, threading
def worker(stream, n):
    x = torch.randn(256, 256, device="cuda:0")
    with torch.cuda.stream(stream):
        for _ in range(n):
            x = x @ x
    stream.synchronize()

streams = [torch.cuda.Stream(device=0) for _ in range(4)]
threads = [threading.Thread(target=worker, args=(s, 50)) for s in streams]
for t in threads: t.start()
for t in threads: t.join()
torch.cuda.synchronize()
print("OK")
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Crashed with: {stderr[-500:]}"
        assert os.path.exists(trace)

    def test_4_streams_captures_all_kernels(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.db")
        script = """
import torch, threading
def worker(stream, n):
    x = torch.randn(256, 256, device="cuda:0")
    with torch.cuda.stream(stream):
        for _ in range(n):
            x = x @ x
    stream.synchronize()

N_STREAMS, N_ITERS = 4, 50
streams = [torch.cuda.Stream(device=0) for _ in range(N_STREAMS)]
threads = [threading.Thread(target=worker, args=(s, N_ITERS)) for s in streams]
for t in threads: t.start()
for t in threads: t.join()
torch.cuda.synchronize()
"""
        rc, _ = _run_traced(script, trace)
        assert rc == 0
        conn = sqlite3.connect(trace)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # 4 streams x 50 matmuls = 200 GEMM ops minimum
        assert count >= 200, f"Expected >=200 ops, got {count}"

    def test_timing_sanity(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.db")
        script = """
import torch, threading
def worker(stream):
    x = torch.randn(256, 256, device="cuda:0")
    with torch.cuda.stream(stream):
        for _ in range(20):
            x = x @ x
    stream.synchronize()

streams = [torch.cuda.Stream(device=0) for _ in range(2)]
threads = [threading.Thread(target=worker, args=(s,)) for s in streams]
for t in threads: t.start()
for t in threads: t.join()
torch.cuda.synchronize()
"""
        rc, _ = _run_traced(script, trace)
        assert rc == 0
        conn = sqlite3.connect(trace)
        bad = conn.execute(
            "SELECT count(*) FROM rocpd_op WHERE end <= start"
        ).fetchone()[0]
        absurd = conn.execute(
            "SELECT count(*) FROM rocpd_op WHERE (end - start) > 60000000000"
        ).fetchone()[0]
        conn.close()
        assert bad == 0, f"{bad} ops with end <= start"
        assert absurd == 0, f"{absurd} ops with duration > 60s"


class TestMultiGPU:
    """Multi-GPU dispatches from multiple threads."""

    def test_2_gpus_no_crash(self, tmp_path):
        _skip_if_no_gpu()
        # Check we have at least 2 GPUs
        r = subprocess.run(
            [sys.executable, "-c", "import torch; assert torch.cuda.device_count() >= 2"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if r.returncode != 0:
            import pytest
            pytest.skip("Need >=2 GPUs")

        trace = str(tmp_path / "trace.db")
        script = """
import torch, threading
def worker(gpu, n):
    x = torch.randn(256, 256, device=f"cuda:{gpu}")
    stream = torch.cuda.Stream(device=gpu)
    with torch.cuda.stream(stream):
        for _ in range(n):
            x = x @ x
    stream.synchronize()

threads = [threading.Thread(target=worker, args=(i, 30)) for i in range(2)]
for t in threads: t.start()
for t in threads: t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
print("OK")
"""
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, f"Crashed: {stderr[-500:]}"

    def test_2_gpus_both_have_ops(self, tmp_path):
        _skip_if_no_gpu()
        r = subprocess.run(
            [sys.executable, "-c", "import torch; assert torch.cuda.device_count() >= 2"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if r.returncode != 0:
            import pytest
            pytest.skip("Need >=2 GPUs")

        trace = str(tmp_path / "trace.db")
        script = """
import torch, threading
def worker(gpu, n):
    x = torch.randn(256, 256, device=f"cuda:{gpu}")
    stream = torch.cuda.Stream(device=gpu)
    with torch.cuda.stream(stream):
        for _ in range(n):
            x = x @ x
    stream.synchronize()

threads = [threading.Thread(target=worker, args=(i, 30)) for i in range(2)]
for t in threads: t.start()
for t in threads: t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _ = _run_traced(script, trace)
        assert rc == 0
        conn = sqlite3.connect(trace)
        gpu_ops = dict(conn.execute(
            "SELECT gpuId, count(*) FROM rocpd_op GROUP BY gpuId"
        ).fetchall())
        conn.close()
        assert 0 in gpu_ops, "No ops on GPU 0"
        assert 1 in gpu_ops, "No ops on GPU 1"
        assert gpu_ops[0] >= 30, f"GPU 0: expected >=30 ops, got {gpu_ops[0]}"
        assert gpu_ops[1] >= 30, f"GPU 1: expected >=30 ops, got {gpu_ops[1]}"


class TestStress:
    """High-volume concurrent dispatch stress test."""

    def test_high_volume_concurrent(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.db")
        script = """
import torch, threading
def worker(stream, n):
    x = torch.randn(128, 128, device="cuda:0")
    with torch.cuda.stream(stream):
        for _ in range(n):
            x = x @ x
    stream.synchronize()

N_STREAMS, N_ITERS = 8, 200
streams = [torch.cuda.Stream(device=0) for _ in range(N_STREAMS)]
threads = [threading.Thread(target=worker, args=(s, N_ITERS)) for s in streams]
for t in threads: t.start()
for t in threads: t.join()
torch.cuda.synchronize()
"""
        rc, stderr = _run_traced(script, trace, timeout=180)
        assert rc == 0, f"Stress test crashed: {stderr[-500:]}"
        conn = sqlite3.connect(trace)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # 8 x 200 = 1600 GEMMs minimum
        assert count >= 1600, f"Expected >=1600, got {count} (data loss?)"

    def test_no_data_loss_under_pressure(self, tmp_path):
        """Verify single completion worker doesn't drop records."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.db")
        # Known exact count: 4 threads x 100 matmuls = 400 GEMMs
        script = """
import torch, threading
def worker(stream):
    x = torch.randn(128, 128, device="cuda:0")
    with torch.cuda.stream(stream):
        for _ in range(100):
            x = x @ x
    stream.synchronize()

streams = [torch.cuda.Stream(device=0) for _ in range(4)]
threads = [threading.Thread(target=worker, args=(s,)) for s in streams]
for t in threads: t.start()
for t in threads: t.join()
torch.cuda.synchronize()
"""
        rc, _ = _run_traced(script, trace)
        assert rc == 0
        conn = sqlite3.connect(trace)
        # Count only GEMM kernels (Cijk)
        gemm_count = conn.execute(
            "SELECT count(*) FROM rocpd_op o JOIN rocpd_string s "
            "ON o.description_id = s.id WHERE s.string LIKE '%Cijk%'"
        ).fetchone()[0]
        conn.close()
        # Should be exactly 400 (4 x 100), allow small tolerance for
        # randn/fill kernels that might also match
        assert gemm_count >= 400, (
            f"Data loss: expected >=400 GEMMs, got {gemm_count}"
        )


class TestCleanShutdown:
    """Verify no hang or crash during shutdown with pending work."""

    def test_exit_during_active_dispatch(self, tmp_path):
        """Process exits while kernels may still be in-flight."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.db")
        # Launch many async dispatches and exit immediately without sync
        script = """
import torch
x = torch.randn(256, 256, device="cuda:0")
for _ in range(50):
    x = x @ x
# NO torch.cuda.synchronize() — exit with work in flight
print("exiting")
"""
        rc, stderr = _run_traced(script, trace, timeout=30)
        # Should not hang or crash
        assert rc == 0, f"Hung or crashed on exit: {stderr[-500:]}"
        assert os.path.exists(trace), "No trace file created"
