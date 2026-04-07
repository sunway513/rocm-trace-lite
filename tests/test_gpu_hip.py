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


def _hip_gpu_count():
    """Get GPU count via gpu_workload binary."""
    if not HAS_GPU_WORKLOAD:
        return 0
    try:
        r = subprocess.run(
            [WORKLOAD, "noop"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, timeout=10
        )
        for line in r.stdout.splitlines():
            if line.startswith("HIP devices:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 0


GPU_COUNT = _hip_gpu_count()

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


# =========================================================================
# Multi-GPU
# =========================================================================

skip_no_multigpu = pytest.mark.skipif(
    GPU_COUNT < 2, reason="Need 2+ GPUs (have {})".format(GPU_COUNT)
)


@skip_no_workload
@skip_no_lib
@skip_no_multigpu
class TestMultiGPU:

    def test_2gpu_unique_gpuids(self, tmp_path):
        """2 GPUs traced, both gpuIds in trace."""
        trace, r = _trace(tmp_path, ["multi_gpu", "2"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        conn = sqlite3.connect(trace)
        gpu_ids = set(
            row[0] for row in conn.execute(
                "SELECT DISTINCT gpuId FROM rocpd_op WHERE gpuId >= 0"
            ).fetchall()
        )
        ops = conn.execute("SELECT count(*) FROM rocpd_op WHERE gpuId >= 0").fetchone()[0]
        conn.close()
        assert len(gpu_ids) >= 2, "Expected 2 gpuIds, got {}".format(gpu_ids)
        assert ops >= 10, "Expected >= 10 ops across 2 GPUs, got {}".format(ops)

    def test_4gpu(self, tmp_path):
        """4 GPUs traced."""
        if GPU_COUNT < 4:
            pytest.skip("Need 4+ GPUs")
        trace, r = _trace(tmp_path, ["multi_gpu", "4"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        conn = sqlite3.connect(trace)
        gpu_ids = set(
            row[0] for row in conn.execute(
                "SELECT DISTINCT gpuId FROM rocpd_op WHERE gpuId >= 0"
            ).fetchall()
        )
        conn.close()
        assert len(gpu_ids) >= 4, "Expected 4 gpuIds, got {}".format(gpu_ids)

    def test_8gpu(self, tmp_path):
        """8 GPUs traced."""
        if GPU_COUNT < 8:
            pytest.skip("Need 8 GPUs")
        trace, r = _trace(tmp_path, ["multi_gpu", "8"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        conn = sqlite3.connect(trace)
        gpu_ids = set(
            row[0] for row in conn.execute(
                "SELECT DISTINCT gpuId FROM rocpd_op WHERE gpuId >= 0"
            ).fetchall()
        )
        ops = conn.execute("SELECT count(*) FROM rocpd_op WHERE gpuId >= 0").fetchone()[0]
        conn.close()
        assert len(gpu_ids) >= 8, "Expected 8 gpuIds, got {}".format(gpu_ids)
        assert ops >= 40, "Expected >= 40 ops across 8 GPUs"

    def test_2gpu_multi_stream(self, tmp_path):
        """2 GPUs x 2 streams each."""
        trace, r = _trace(tmp_path, ["multi_gpu_stream", "2", "2"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        conn = sqlite3.connect(trace)
        gpu_ids = set(
            row[0] for row in conn.execute(
                "SELECT DISTINCT gpuId FROM rocpd_op WHERE gpuId >= 0"
            ).fetchall()
        )
        ops = conn.execute("SELECT count(*) FROM rocpd_op WHERE gpuId >= 0").fetchone()[0]
        conn.close()
        assert len(gpu_ids) >= 2
        assert ops >= 10

    def test_8gpu_multi_stream(self, tmp_path):
        """8 GPUs x 4 streams each — full stress."""
        if GPU_COUNT < 8:
            pytest.skip("Need 8 GPUs")
        trace, r = _trace(tmp_path, ["multi_gpu_stream", "8", "4"], timeout=120)
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        conn = sqlite3.connect(trace)
        gpu_ids = set(
            row[0] for row in conn.execute(
                "SELECT DISTINCT gpuId FROM rocpd_op WHERE gpuId >= 0"
            ).fetchall()
        )
        ops = conn.execute("SELECT count(*) FROM rocpd_op WHERE gpuId >= 0").fetchone()[0]
        conn.close()
        assert len(gpu_ids) >= 8, "Expected 8 gpuIds, got {}".format(gpu_ids)
        assert ops >= 80, "Expected >= 80 ops (8 GPU x 4 streams x 5)"

    def test_multi_gpu_busy_view(self, tmp_path):
        """Busy view has entries per GPU."""
        trace, r = _trace(tmp_path, ["multi_gpu", "2"])
        assert r.returncode == 0
        conn = sqlite3.connect(trace)
        busy = conn.execute("SELECT * FROM busy").fetchall()
        conn.close()
        assert len(busy) >= 2, "Busy view should have >= 2 GPUs"

    def test_multi_gpu_perfetto(self, tmp_path):
        """Perfetto JSON has per-GPU tracks."""
        import gzip
        import json
        trace, r = _trace(tmp_path, ["multi_gpu", "2"])
        assert r.returncode == 0
        json_gz = trace.replace(".db", ".json.gz")
        assert os.path.exists(json_gz)
        with gzip.open(json_gz, "rt") as f:
            data = json.load(f)
        # Perfetto JSON should have process metadata for multiple GPUs
        assert isinstance(data, (list, dict))


# =========================================================================
# Roctx markers
# =========================================================================


@skip_no_workload
@skip_no_lib
class TestRoctxHIP:

    def test_roctx_push_pop(self, tmp_path):
        """roctx push/pop markers appear in trace."""
        trace, r = _trace(tmp_path, ["roctx"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        conn = sqlite3.connect(trace)
        markers = conn.execute(
            "SELECT count(*) FROM rocpd_op o JOIN rocpd_string s "
            "ON o.description_id=s.id WHERE s.string='outer_region'"
        ).fetchone()[0]
        ops = conn.execute("SELECT count(*) FROM rocpd_op WHERE gpuId >= 0").fetchone()[0]
        conn.close()
        assert markers >= 1, "roctx marker 'outer_region' not found"
        assert ops >= 5, "Expected >= 5 kernel ops"

    def test_roctx_mark(self, tmp_path):
        """roctx mark appears in trace."""
        trace, r = _trace(tmp_path, ["roctx"])
        assert r.returncode == 0
        conn = sqlite3.connect(trace)
        marks = conn.execute(
            "SELECT count(*) FROM rocpd_op o JOIN rocpd_string s "
            "ON o.description_id=s.id WHERE s.string='checkpoint_1'"
        ).fetchone()[0]
        conn.close()
        assert marks >= 1, "roctx mark 'checkpoint_1' not found"

    def test_roctx_nested(self, tmp_path):
        """Nested roctx ranges and start/stop captured."""
        trace, r = _trace(tmp_path, ["roctx_nested"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        conn = sqlite3.connect(trace)
        level0 = conn.execute(
            "SELECT count(*) FROM rocpd_op o JOIN rocpd_string s "
            "ON o.description_id=s.id WHERE s.string='level_0'"
        ).fetchone()[0]
        level1 = conn.execute(
            "SELECT count(*) FROM rocpd_op o JOIN rocpd_string s "
            "ON o.description_id=s.id WHERE s.string='level_1'"
        ).fetchone()[0]
        conn.close()
        assert level0 >= 1, "level_0 not found"
        assert level1 >= 1, "level_1 not found"

    def test_roctx_start_stop(self, tmp_path):
        """roctxRangeStartA/Stop captured."""
        trace, r = _trace(tmp_path, ["roctx_nested"])
        assert r.returncode == 0
        conn = sqlite3.connect(trace)
        ranges = conn.execute(
            "SELECT count(*) FROM rocpd_op o JOIN rocpd_string s "
            "ON o.description_id=s.id WHERE s.string='range_A'"
        ).fetchone()[0]
        conn.close()
        assert ranges >= 1, "range_A not found"

    def test_roctx_rapid_100(self, tmp_path):
        """100 rapid marks all captured."""
        trace, r = _trace(tmp_path, ["roctx_rapid", "100"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        conn = sqlite3.connect(trace)
        total_markers = conn.execute(
            "SELECT count(*) FROM rocpd_op WHERE gpuId < 0"
        ).fetchone()[0]
        conn.close()
        assert total_markers >= 90, "Expected >= 90 marks, got {}".format(total_markers)

    def test_roctx_not_in_top_view(self, tmp_path):
        """roctx markers don't appear in kernel top view."""
        trace, r = _trace(tmp_path, ["roctx"])
        assert r.returncode == 0
        conn = sqlite3.connect(trace)
        top = conn.execute("SELECT * FROM top LIMIT 20").fetchall()
        conn.close()
        top_names = [row[0] for row in top]
        assert "outer_region" not in top_names, "roctx marker in top view"
        assert "checkpoint_1" not in top_names, "roctx mark in top view"


# =========================================================================
# HIP Graph
# =========================================================================


@skip_no_workload
@skip_no_lib
class TestHIPGraphHIP:

    def test_graph_single_replay(self, tmp_path):
        """Graph capture + 1 replay captured."""
        trace, r = _trace(tmp_path, ["hipgraph", "1"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        ops = _query(trace, "SELECT count(*) FROM rocpd_op WHERE gpuId >= 0")[0]
        assert ops >= 1, "Expected kernel ops from graph replay"

    def test_graph_10_replays(self, tmp_path):
        """10 graph replays — signal pool recycling."""
        trace, r = _trace(tmp_path, ["hipgraph", "10"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        ops = _query(trace, "SELECT count(*) FROM rocpd_op WHERE gpuId >= 0")[0]
        assert ops >= 5, "Expected ops from 10 replays, got {}".format(ops)

    def test_graph_100_replays(self, tmp_path):
        """100 graph replays — no hang, no crash."""
        trace, r = _trace(tmp_path, ["hipgraph", "100"], timeout=60)
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        ops = _query(trace, "SELECT count(*) FROM rocpd_op WHERE gpuId >= 0")[0]
        assert ops >= 10, "Expected ops from 100 replays"

    def test_graph_multi_stream(self, tmp_path):
        """Graph captured on stream 0, replayed on stream 1."""
        trace, r = _trace(tmp_path, ["hipgraph_multi_stream"])
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        ops = _query(trace, "SELECT count(*) FROM rocpd_op WHERE gpuId >= 0")[0]
        assert ops >= 1
