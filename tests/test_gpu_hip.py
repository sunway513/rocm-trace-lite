"""GPU E2E tests using HIP workload binary — no PyTorch dependency.

Requires: tests/gpu_workload compiled via `make tests/gpu_workload` (hipcc).
Run with: python3 -m pytest tests/test_gpu_hip.py -v --timeout=120
"""
import os
import sqlite3
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "rocm_trace_lite", "lib", "librtl.so")
LIB_BUILD = os.path.join(REPO_ROOT, "librtl.so")
WORKLOAD = os.path.join(REPO_ROOT, "tests", "gpu_workload")

HAS_GPU_WORKLOAD = os.path.exists(WORKLOAD)
HAS_LIB = os.path.exists(LIB_PATH) or os.path.exists(LIB_BUILD)

skip_no_workload = pytest.mark.skipif(
    not HAS_GPU_WORKLOAD, reason="gpu_workload not compiled (run: make tests/gpu_workload)"
)
skip_no_lib = pytest.mark.skipif(
    not HAS_LIB, reason="librtl.so not built"
)


def _lib():
    return LIB_PATH if os.path.exists(LIB_PATH) else LIB_BUILD


def _trace(tmp_path, args, name="trace", timeout=60):
    """Run gpu_workload under rtl trace, return (trace_path, result)."""
    trace = str(tmp_path / "{}.db".format(name))
    env = os.environ.copy()
    env["PYTHONPATH"] = "{}:{}".format(REPO_ROOT, env.get("PYTHONPATH", ""))
    cmd = [
        sys.executable, "-m", "rocm_trace_lite.cli", "trace", "-o", trace,
        WORKLOAD,
    ] + args
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=timeout, cwd=REPO_ROOT, env=env,
    )
    return trace, result


def _query(db, sql):
    conn = sqlite3.connect(db)
    val = conn.execute(sql).fetchone()
    conn.close()
    return val


# =========================================================================
# Basic tracing
# =========================================================================


@skip_no_workload
@skip_no_lib
class TestBasicTracing:

    def test_gemm_captured(self, tmp_path):
        """GEMM dispatches are captured in trace."""
        trace, r = _trace(tmp_path, ["gemm", "128", "10"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        ops = _query(trace, "SELECT count(*) FROM rocpd_op")[0]
        assert ops >= 10, "Expected >= 10 ops, got {}".format(ops)

    def test_short_kernels(self, tmp_path):
        """1000 short kernels captured without hang."""
        trace, r = _trace(tmp_path, ["short", "1000"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        ops = _query(trace, "SELECT count(*) FROM rocpd_op")[0]
        assert ops >= 500, "Expected >= 500 ops, got {} (some drops OK)".format(ops)

    def test_noop_no_crash(self, tmp_path):
        """No kernels dispatched — profiler doesn't crash."""
        trace, r = _trace(tmp_path, ["noop"])
        # May return non-zero (0 ops warning) but must not crash/timeout
        assert "rtl:" in r.stderr or r.returncode == 0

    def test_summary_output(self, tmp_path):
        """rtl trace stdout has kernel summary."""
        trace, r = _trace(tmp_path, ["gemm", "256", "20"])
        assert r.returncode == 0
        assert "GPU ops" in r.stdout

    def test_perfetto_json(self, tmp_path):
        """Produces .json.gz Perfetto file."""
        import gzip
        import json
        trace, r = _trace(tmp_path, ["gemm", "128", "10"])
        assert r.returncode == 0
        json_gz = trace.replace(".db", ".json.gz")
        assert os.path.exists(json_gz)
        with gzip.open(json_gz, "rt") as f:
            data = json.load(f)
        assert isinstance(data, (list, dict))

    def test_kernel_names_not_hex(self, tmp_path):
        """Kernel names are demangled, not hex addresses."""
        trace, r = _trace(tmp_path, ["gemm", "64", "5"])
        assert r.returncode == 0
        conn = sqlite3.connect(trace)
        names = [row[0] for row in conn.execute(
            "SELECT s.string FROM rocpd_op o JOIN rocpd_string s "
            "ON o.description_id=s.id WHERE o.gpuId >= 0"
        ).fetchall()]
        conn.close()
        hex_only = [n for n in names if n.startswith("kernel_0x")]
        assert len(hex_only) < len(names), "All names are hex fallback"

    def test_top_view(self, tmp_path):
        """Top view returns sorted kernel stats."""
        trace, r = _trace(tmp_path, ["gemm", "128", "50"])
        assert r.returncode == 0
        conn = sqlite3.connect(trace)
        top = conn.execute("SELECT * FROM top LIMIT 5").fetchall()
        conn.close()
        assert len(top) > 0

    def test_busy_view(self, tmp_path):
        """Busy view returns GPU utilization."""
        trace, r = _trace(tmp_path, ["gemm", "128", "10"])
        assert r.returncode == 0
        conn = sqlite3.connect(trace)
        busy = conn.execute("SELECT * FROM busy").fetchall()
        conn.close()
        assert len(busy) >= 1


# =========================================================================
# Multi-stream / multi-thread
# =========================================================================


@skip_no_workload
@skip_no_lib
class TestConcurrency:

    def test_multi_stream_4(self, tmp_path):
        """4 streams traced, multiple queueIds."""
        trace, r = _trace(tmp_path, ["multi_stream", "4"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        ops = _query(trace, "SELECT count(*) FROM rocpd_op")[0]
        assert ops >= 20, "Expected >= 20 ops across 4 streams"

    def test_multi_stream_8(self, tmp_path):
        """8 streams traced."""
        trace, r = _trace(tmp_path, ["multi_stream", "8"])
        assert r.returncode == 0
        ops = _query(trace, "SELECT count(*) FROM rocpd_op")[0]
        assert ops >= 40

    def test_multi_thread_4(self, tmp_path):
        """4 threads traced."""
        trace, r = _trace(tmp_path, ["multi_thread", "4"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        ops = _query(trace, "SELECT count(*) FROM rocpd_op")[0]
        assert ops >= 10

    def test_multi_thread_8(self, tmp_path):
        """8 threads traced."""
        trace, r = _trace(tmp_path, ["multi_thread", "8"])
        assert r.returncode == 0
        ops = _query(trace, "SELECT count(*) FROM rocpd_op")[0]
        assert ops >= 20


# =========================================================================
# Stress
# =========================================================================


@skip_no_workload
@skip_no_lib
class TestStressHIP:

    def test_10k_short_kernels(self, tmp_path):
        """10K tiny kernels — signal pool recycling."""
        trace, r = _trace(tmp_path, ["short", "10000"], timeout=120)
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        ops = _query(trace, "SELECT count(*) FROM rocpd_op")[0]
        assert ops >= 8000, "Too many drops: {} ops from 10K".format(ops)

    def test_large_gemm(self, tmp_path):
        """100 large GEMM (512x512)."""
        trace, r = _trace(tmp_path, ["gemm", "512", "100"], timeout=120)
        assert r.returncode == 0
        ops = _query(trace, "SELECT count(*) FROM rocpd_op")[0]
        assert ops >= 90

    def test_diagnostic_counters(self, tmp_path):
        """Diagnostic output shows injected + recorded counts."""
        trace, r = _trace(tmp_path, ["gemm", "128", "50"])
        assert r.returncode == 0
        assert "rtl diagnostic" in r.stderr or "rtl: trace finalized" in r.stderr


# =========================================================================
# Overhead
# =========================================================================


@skip_no_workload
@skip_no_lib
class TestOverheadHIP:

    def test_trace_file_size_reasonable(self, tmp_path):
        """1000 ops < 5MB trace file."""
        trace, r = _trace(tmp_path, ["gemm", "64", "1000"])
        assert r.returncode == 0
        size_mb = os.path.getsize(trace) / 1024 / 1024
        assert size_mb < 5, "Trace too large: {:.1f}MB".format(size_mb)
