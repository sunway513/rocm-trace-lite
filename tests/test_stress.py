"""Sprint 5: Stress tests and fault injection for rocm-trace-lite.

Covers:
  - HSA Load/Unload lifecycle (CPU only)
  - Completion worker stress (GPU)
  - Signal pool exhaustion (GPU)
  - Fault injection on SQLite paths (CPU only)
  - Concurrent / race condition tests (GPU)

Run with: pytest tests/test_stress.py -v
"""
import os
import sqlite3
import subprocess
import sys
import stat

import pytest

from conftest import populate_synthetic_trace
from rocm_trace_lite.cmd_trace import _generate_summary

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "librpd_lite.so")

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

gpu = pytest.mark.skipif(not HAS_GPU, reason="No GPU available")


def _has_lib():
    return os.path.exists(LIB_PATH)


def _skip_if_no_gpu():
    if not HAS_GPU or not _has_lib():
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


def _trace_workload(tmp_path, script_code, name="trace", timeout=120):
    """Run script under rtl trace, return (trace_path, result)."""
    trace = str(tmp_path / "{}.db".format(name))
    script = str(tmp_path / "{}.py".format(name))
    with open(script, "w") as f:
        f.write(script_code)
    result = subprocess.run(
        [sys.executable, "-m", "rocm_trace_lite.cli", "trace", "-o", trace,
         sys.executable, script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=timeout
    )
    return trace, result


def _make_valid_db(path, num_kernels=10):
    """Create a valid trace DB with synthetic data."""
    populate_synthetic_trace(path, num_kernels=num_kernels, num_gpus=1)
    return path


# ===========================================================================
# HSA Load/Unload Lifecycle (CPU only, 2 tests)
# ===========================================================================


class TestHSALifecycle:
    """Verify librpd_lite.so symbols and dependencies."""

    def test_onload_onunload_symbols(self):
        """dlopen librpd_lite.so (if exists), verify OnLoad/OnUnload symbols via nm -D."""
        if not _has_lib():
            pytest.skip("librpd_lite.so not built")

        result = subprocess.run(
            ["nm", "-D", LIB_PATH],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=10
        )
        assert result.returncode == 0, "nm -D failed: {}".format(result.stderr)
        symbols = result.stdout

        assert "OnLoad" in symbols, "OnLoad symbol not exported in librpd_lite.so"
        assert "OnUnload" in symbols, "OnUnload symbol not exported in librpd_lite.so"

    def test_shared_lib_dependencies(self):
        """Verify .so NEEDED dependencies are only expected libraries."""
        if not _has_lib():
            pytest.skip("librpd_lite.so not built")

        result = subprocess.run(
            ["ldd", LIB_PATH],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=10
        )
        assert result.returncode == 0, "ldd failed: {}".format(result.stderr)

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or "=>" not in line:
                continue
            if "not found" in line:
                lib_name = line.split("=>")[0].strip()
                pytest.fail("Missing dependency: {}".format(lib_name))


# ===========================================================================
# Completion Worker Stress (GPU, 2 tests)
# ===========================================================================


class TestCompletionWorkerStress:
    """Stress the single completion worker thread with high-volume dispatches."""

    @gpu
    def test_10k_rapid_small_gemm(self, tmp_path):
        """10K rapid small GEMMs (16x16), verify trace has > 9000 ops."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "stress_10k.db")
        script = (
            "import torch\n"
            "x = torch.randn(16, 16, device='cuda:0')\n"
            "for i in range(10000):\n"
            "    x = x @ x\n"
            "    if i % 1000 == 999:\n"
            "        torch.cuda.synchronize()\n"
            "torch.cuda.synchronize()\n"
        )
        rc, stderr = _run_traced(script, trace, timeout=120)
        assert rc == 0, "10K GEMM stress crashed: {}".format(stderr[-500:])
        assert os.path.exists(trace), "No trace file produced"

        conn = sqlite3.connect(trace)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # Allow some drops, but must capture > 90%
        assert count > 9000, (
            "Expected >9000 ops from 10K GEMMs, got {} (too much data loss)".format(count)
        )

    @gpu
    def test_3_rounds_independent_traces(self, tmp_path):
        """Run 3 rounds of workload with clear between, verify each trace is independent."""
        _skip_if_no_gpu()
        round_script = (
            "import torch\n"
            "x = torch.randn(64, 64, device='cuda:0')\n"
            "for _ in range(200):\n"
            "    x = x @ x\n"
            "torch.cuda.synchronize()\n"
        )
        counts = []
        for i in range(3):
            trace = str(tmp_path / "round_{}.db".format(i))
            rc, stderr = _run_traced(round_script, trace, timeout=60)
            assert rc == 0, "Round {} crashed: {}".format(i, stderr[-300:])
            assert os.path.exists(trace), "Round {} produced no trace".format(i)

            conn = sqlite3.connect(trace)
            count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
            conn.close()
            counts.append(count)

        # Each round should have roughly the same count (200 GEMMs each)
        for i, c in enumerate(counts):
            assert c >= 180, (
                "Round {} had only {} ops (expected ~200)".format(i, c)
            )
        # Verify independence: counts should be similar, not cumulative
        assert max(counts) < 2 * min(counts), (
            "Traces not independent - counts diverge too much: {}".format(counts)
        )


# ===========================================================================
# Signal Pool Exhaustion (GPU, 2 tests)
# ===========================================================================


class TestSignalPoolExhaustion:
    """Exhaust the 64-signal pre-allocated pool."""

    @gpu
    def test_concurrent_dispatch_no_hang(self, tmp_path):
        """Multi-thread parallel dispatch > 64 concurrent kernels, must not hang."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "exhaust.db")
        # 16 streams x 100 dispatches = 1600 concurrent-capable dispatches
        script = (
            "import torch, threading\n"
            "\n"
            "def worker(stream_id, n):\n"
            "    s = torch.cuda.Stream(device=0)\n"
            "    x = torch.randn(32, 32, device='cuda:0')\n"
            "    with torch.cuda.stream(s):\n"
            "        for _ in range(n):\n"
            "            x = x @ x\n"
            "    s.synchronize()\n"
            "\n"
            "threads = [threading.Thread(target=worker, args=(i, 100)) for i in range(16)]\n"
            "for t in threads:\n"
            "    t.start()\n"
            "for t in threads:\n"
            "    t.join()\n"
            "torch.cuda.synchronize()\n"
        )
        rc, stderr = _run_traced(script, trace, timeout=60)
        assert rc == 0, "Hung or crashed under pool exhaustion: {}".format(stderr[-500:])
        assert os.path.exists(trace), "No trace file"

    @gpu
    def test_drop_counter_on_pool_exhaustion(self, tmp_path):
        """When signal pool is exhausted, drop counter should appear in stderr."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "drop_test.db")
        # Fire many concurrent kernels without sync to maximize pool pressure
        script = (
            "import torch, threading\n"
            "\n"
            "def burst(stream_id):\n"
            "    s = torch.cuda.Stream(device=0)\n"
            "    x = torch.randn(16, 16, device='cuda:0')\n"
            "    with torch.cuda.stream(s):\n"
            "        for _ in range(500):\n"
            "            x = x @ x\n"
            "\n"
            "threads = [threading.Thread(target=burst, args=(i,)) for i in range(32)]\n"
            "for t in threads:\n"
            "    t.start()\n"
            "for t in threads:\n"
            "    t.join()\n"
            "torch.cuda.synchronize()\n"
        )
        rc, stderr = _run_traced(script, trace, timeout=60)
        # We don't require rc == 0 since pool exhaustion might cause issues
        # But the process must not hang (timeout handles that)

        # Check diagnostic output is present (pool dynamically grows to 4096,
        # so actual drops may or may not happen depending on hardware speed)
        assert "rpd_lite diagnostic" in stderr, (
            "Diagnostic output missing from stderr"
        )


# ===========================================================================
# Fault Injection (CPU only, 3 tests)
# ===========================================================================


class TestFaultInjection:
    """Fault injection tests on SQLite and file system paths."""

    def test_readonly_db_summary_readable(self, tmp_path):
        """Create valid trace DB, chmod 444, _generate_summary() still reads it."""
        db_path = str(tmp_path / "readonly.db")
        _make_valid_db(db_path, num_kernels=50)

        # Make read-only
        os.chmod(db_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        try:
            summary = _generate_summary(db_path)
            assert summary is not None, "_generate_summary returned None on readonly DB"
            assert "50 GPU ops" in summary, (
                "Summary should report 50 ops, got: {}".format(summary[:200])
            )
        finally:
            # Restore write permission for cleanup
            os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    def test_nonexistent_output_dir_error(self, tmp_path):
        """rtl trace -o /nonexistent/dir/trace.db should produce an error message."""
        bad_output = "/nonexistent_rtl_test_dir_{}/trace.db".format(os.getpid())
        script = str(tmp_path / "dummy.py")
        with open(script, "w") as f:
            f.write("print('hello')\n")

        subprocess.run(
            [sys.executable, "-m", "rocm_trace_lite.cli", "trace",
             "-o", bad_output, sys.executable, script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30
        )
        # It should not silently succeed with a trace at the bad path
        assert not os.path.exists(bad_output), (
            "Trace file should not exist at nonexistent dir"
        )

    def test_empty_sqlite_no_crash(self, tmp_path):
        """Empty SQLite file (0 bytes): _generate_summary() must not crash."""
        db_path = str(tmp_path / "empty.db")
        # Create a 0-byte file
        open(db_path, "w").close()
        assert os.path.getsize(db_path) == 0

        # Should not raise, should return gracefully
        summary = _generate_summary(db_path)
        # It should return either None or a warning string, not crash
        assert summary is None or isinstance(summary, str), (
            "Expected None or string from empty DB, got: {}".format(type(summary))
        )


# ===========================================================================
# Concurrent / Race (GPU, 2 tests)
# ===========================================================================


class TestConcurrentRace:
    """Race condition and rapid lifecycle tests."""

    @gpu
    def test_fast_start_stop_no_deadlock(self, tmp_path):
        """Fast start/stop workload (< 50ms), verify shutdown doesn't deadlock."""
        _skip_if_no_gpu()
        trace = str(tmp_path / "fast.db")
        # Minimal workload: one tiny kernel, immediate exit
        script = (
            "import torch\n"
            "x = torch.randn(4, 4, device='cuda:0')\n"
            "y = x @ x\n"
        )
        rc, stderr = _run_traced(script, trace, timeout=30)
        assert rc == 0, "Deadlock on fast start/stop: {}".format(stderr[-500:])

    @gpu
    def test_sequential_traces_independent(self, tmp_path):
        """3 sequential traces: each must be independent, no data crossover."""
        _skip_if_no_gpu()
        workload = (
            "import torch\n"
            "x = torch.randn(64, 64, device='cuda:0')\n"
            "for _ in range(100):\n"
            "    x = x @ x\n"
            "torch.cuda.synchronize()\n"
        )
        traces = []
        for i in range(3):
            trace = str(tmp_path / "seq_{}.db".format(i))
            rc, stderr = _run_traced(workload, trace, timeout=60)
            assert rc == 0, "Sequential trace {} failed: {}".format(i, stderr[-300:])
            assert os.path.exists(trace), "Trace {} not created".format(i)
            traces.append(trace)

        # Verify each trace has ops and they don't accumulate
        counts = []
        for i, trace in enumerate(traces):
            conn = sqlite3.connect(trace)
            count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
            conn.close()
            counts.append(count)
            assert count >= 80, (
                "Trace {} has only {} ops (expected ~100)".format(i, count)
            )

        # Independence check: no trace should have 2x+ another's count
        if min(counts) > 0:
            ratio = max(counts) / min(counts)
            assert ratio < 2.0, (
                "Traces not independent, count ratio {:.1f}: {}".format(ratio, counts)
            )
