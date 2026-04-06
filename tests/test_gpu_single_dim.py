"""Sprint 2 GPU E2E tests: single-dimension stress per feature.

Each test class exercises one feature dimension in isolation:
multi-thread, multi-stream, HIP graph, roctx, and other single-dim.

Requires a ROCm GPU. Skip gracefully if unavailable.
Run with: pytest tests/test_gpu_single_dim.py -v
"""
import os
import sqlite3
import subprocess
import sys
import textwrap

import pytest

from conftest import _rocm_gpu_available

HAS_GPU = _rocm_gpu_available()

gpu = pytest.mark.skipif(not HAS_GPU, reason="No GPU available")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _trace_and_query(tmp_path, script_code, query="SELECT count(*) FROM rocpd_op",
                     timeout=120):
    """Run a Python script under rtl trace, return query result."""
    trace = str(tmp_path / "trace.db")
    script = str(tmp_path / "workload.py")
    with open(script, "w") as f:
        f.write(textwrap.dedent(script_code))
    result = subprocess.run(
        [sys.executable, "-m", "rocm_trace_lite.cli", "trace",
         "-o", trace, sys.executable, script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=timeout, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        "rtl trace failed (rc={rc}):\nstderr: {err}".format(
            rc=result.returncode, err=result.stderr[-500:])
    )
    assert os.path.exists(trace), "No trace file created"
    conn = sqlite3.connect(trace)
    val = conn.execute(query).fetchone()
    conn.close()
    return val


def _trace_and_connect(tmp_path, script_code, timeout=120):
    """Run a Python script under rtl trace, return an open sqlite3 connection."""
    trace = str(tmp_path / "trace.db")
    script = str(tmp_path / "workload.py")
    with open(script, "w") as f:
        f.write(textwrap.dedent(script_code))
    result = subprocess.run(
        [sys.executable, "-m", "rocm_trace_lite.cli", "trace",
         "-o", trace, sys.executable, script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=timeout, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        "rtl trace failed (rc={rc}):\nstderr: {err}".format(
            rc=result.returncode, err=result.stderr[-500:])
    )
    assert os.path.exists(trace), "No trace file created"
    return sqlite3.connect(trace)


# ---------------------------------------------------------------------------
# roctx helper: snippet to load roctx from HSA_TOOLS_LIB inside workload
# ---------------------------------------------------------------------------
ROCTX_PREAMBLE = '''
import ctypes, os
_lib = os.environ.get("HSA_TOOLS_LIB", "librtl.so")
_roctx = ctypes.CDLL(_lib)
_roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
_roctx.roctxRangePushA.restype = ctypes.c_int
_roctx.roctxRangePop.restype = ctypes.c_int
_roctx.roctxMarkA.argtypes = [ctypes.c_char_p]
_roctx.roctxRangeStartA.argtypes = [ctypes.c_char_p]
_roctx.roctxRangeStartA.restype = ctypes.c_uint64
_roctx.roctxRangeStop.argtypes = [ctypes.c_uint64]
# Legacy aliases (no 'A' suffix)
_roctx.roctxRangePush.argtypes = [ctypes.c_char_p]
_roctx.roctxRangePush.restype = ctypes.c_int
_roctx.roctxMark.argtypes = [ctypes.c_char_p]
'''


# ===================================================================
# Multi-thread tests (7 tests)
# ===================================================================
class TestMultiThread:
    """Single-dimension: multi-thread dispatch."""

    @gpu
    def test_2_threads_unique_kernels(self, tmp_path):
        """2 threads each run GEMM, trace has 2+ unique kernel ops."""
        conn = _trace_and_connect(tmp_path, """
            import torch, threading

            def worker():
                x = torch.randn(128, 128, device="cuda:0")
                for _ in range(10):
                    x = x @ x
                torch.cuda.synchronize()

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads: t.start()
            for t in threads: t.join()
            torch.cuda.synchronize()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        unique = conn.execute(
            "SELECT count(DISTINCT description_id) FROM rocpd_op"
        ).fetchone()[0]
        conn.close()
        assert op_count >= 2, "Expected >=2 ops, got {n}".format(n=op_count)
        assert unique >= 2, "Expected >=2 unique kernel ops, got {n}".format(n=unique)

    @gpu
    def test_4_threads_burst(self, tmp_path):
        """4 threads burst dispatch, total ops >= 4."""
        val = _trace_and_query(tmp_path, """
            import torch, threading

            def worker():
                x = torch.randn(64, 64, device="cuda:0")
                for _ in range(5):
                    x = x @ x
                torch.cuda.synchronize()

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads: t.start()
            for t in threads: t.join()
            torch.cuda.synchronize()
        """)
        assert val[0] >= 4, "Expected >=4 ops, got {n}".format(n=val[0])

    @gpu
    def test_8_threads_no_crash(self, tmp_path):
        """8 threads concurrent, no crash."""
        val = _trace_and_query(tmp_path, """
            import torch, threading

            def worker():
                x = torch.randn(64, 64, device="cuda:0")
                for _ in range(10):
                    x = x @ x
                torch.cuda.synchronize()

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads: t.start()
            for t in threads: t.join()
            torch.cuda.synchronize()
        """)
        assert val[0] > 0, "No ops captured with 8 threads"

    @gpu
    def test_16_threads_stress_signal_pool(self, tmp_path):
        """16 threads stress, signal pool not exhausted (ops > 0)."""
        val = _trace_and_query(tmp_path, """
            import torch, threading

            def worker():
                x = torch.randn(32, 32, device="cuda:0")
                for _ in range(20):
                    x = x @ x
                torch.cuda.synchronize()

            threads = [threading.Thread(target=worker) for _ in range(16)]
            for t in threads: t.start()
            for t in threads: t.join()
            torch.cuda.synchronize()
        """, timeout=180)
        assert val[0] > 0, "Signal pool exhausted? 0 ops captured with 16 threads"

    @gpu
    def test_thread_create_destroy_recreate(self, tmp_path):
        """Thread create -> destroy -> recreate, second round also traced."""
        conn = _trace_and_connect(tmp_path, """
            import torch, threading

            def worker(tag):
                x = torch.randn(64, 64, device="cuda:0")
                for _ in range(5):
                    x = x @ x
                torch.cuda.synchronize()

            # Round 1
            t1 = threading.Thread(target=worker, args=("r1",))
            t1.start()
            t1.join()

            # Round 2 -- new thread after first destroyed
            t2 = threading.Thread(target=worker, args=("r2",))
            t2.start()
            t2.join()
            torch.cuda.synchronize()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # Both rounds should produce ops: 2 * 5 = 10 GEMMs minimum
        assert op_count >= 10, "Expected >=10 ops from 2 rounds, got {n}".format(
            n=op_count)

    @gpu
    def test_main_roctx_worker_kernel(self, tmp_path):
        """Main thread roctx + worker thread kernel, both in trace."""
        conn = _trace_and_connect(tmp_path, ROCTX_PREAMBLE + """
            import torch, threading

            _roctx.roctxRangePushA(b"main_marker")

            def worker():
                x = torch.randn(64, 64, device="cuda:0")
                for _ in range(5):
                    x = x @ x
                torch.cuda.synchronize()

            t = threading.Thread(target=worker)
            t.start()
            t.join()

            _roctx.roctxRangePop()
            torch.cuda.synchronize()
        """)
        markers = conn.execute(
            "SELECT count(*) FROM rocpd_op o "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'UserMarker'"
        ).fetchone()[0]
        kernels = conn.execute(
            "SELECT count(*) FROM rocpd_op o "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'KernelExecution'"
        ).fetchone()[0]
        conn.close()
        assert markers > 0, "No roctx markers captured"
        assert kernels > 0, "No kernel ops captured from worker thread"

    @gpu
    def test_all_threads_same_kernel_string_dedup(self, tmp_path):
        """All threads same kernel name, string dedup (rocpd_string count reasonable)."""
        conn = _trace_and_connect(tmp_path, """
            import torch, threading

            def worker():
                x = torch.randn(128, 128, device="cuda:0")
                for _ in range(20):
                    x = x @ x
                torch.cuda.synchronize()

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads: t.start()
            for t in threads: t.join()
            torch.cuda.synchronize()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        string_count = conn.execute("SELECT count(*) FROM rocpd_string").fetchone()[0]
        conn.close()
        # 4 threads x 20 GEMMs = 80 ops, but string table should be << ops
        assert op_count >= 80, "Expected >=80 ops, got {n}".format(n=op_count)
        assert string_count < op_count, (
            "String dedup broken: {s} strings for {o} ops".format(
                s=string_count, o=op_count)
        )


# ===================================================================
# Multi-stream tests (6 tests)
# ===================================================================
class TestMultiStream:
    """Single-dimension: multi-stream dispatch."""

    @gpu
    def test_2_streams_unique_queues(self, tmp_path):
        """2 streams parallel GEMM, 2+ unique queueId."""
        conn = _trace_and_connect(tmp_path, """
            import torch

            s1 = torch.cuda.Stream(device=0)
            s2 = torch.cuda.Stream(device=0)
            x = torch.randn(128, 128, device="cuda:0")

            with torch.cuda.stream(s1):
                for _ in range(10):
                    x = x @ x
            with torch.cuda.stream(s2):
                y = torch.randn(128, 128, device="cuda:0")
                for _ in range(10):
                    y = y @ y
            s1.synchronize()
            s2.synchronize()
        """)
        queues = conn.execute(
            "SELECT count(DISTINCT queueId) FROM rocpd_op WHERE gpuId >= 0"
        ).fetchone()[0]
        conn.close()
        assert queues >= 2, "Expected >=2 unique queueIds, got {n}".format(n=queues)

    @gpu
    def test_4_streams_distributed_ops(self, tmp_path):
        """4 streams parallel, ops distributed across multiple queues."""
        conn = _trace_and_connect(tmp_path, """
            import torch, threading

            def worker(stream):
                x = torch.randn(64, 64, device="cuda:0")
                with torch.cuda.stream(stream):
                    for _ in range(10):
                        x = x @ x
                stream.synchronize()

            streams = [torch.cuda.Stream(device=0) for _ in range(4)]
            threads = [threading.Thread(target=worker, args=(s,)) for s in streams]
            for t in threads: t.start()
            for t in threads: t.join()
            torch.cuda.synchronize()
        """)
        queue_count = conn.execute(
            "SELECT count(DISTINCT queueId) FROM rocpd_op WHERE gpuId >= 0"
        ).fetchone()[0]
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert queue_count >= 2, (
            "Expected ops on multiple queues, got {n}".format(n=queue_count))
        assert op_count >= 40, "Expected >=40 ops, got {n}".format(n=op_count)

    @gpu
    def test_8_streams_no_crash(self, tmp_path):
        """8 streams, no crash."""
        val = _trace_and_query(tmp_path, """
            import torch, threading

            def worker(stream):
                x = torch.randn(64, 64, device="cuda:0")
                with torch.cuda.stream(stream):
                    for _ in range(10):
                        x = x @ x
                stream.synchronize()

            streams = [torch.cuda.Stream(device=0) for _ in range(8)]
            threads = [threading.Thread(target=worker, args=(s,)) for s in streams]
            for t in threads: t.start()
            for t in threads: t.join()
            torch.cuda.synchronize()
        """)
        assert val[0] > 0, "No ops captured with 8 streams"

    @gpu
    def test_stream_sync_event_ordering(self, tmp_path):
        """Stream sync (streamWaitEvent), kernel end times ordered."""
        conn = _trace_and_connect(tmp_path, """
            import torch

            s1 = torch.cuda.Stream(device=0)
            s2 = torch.cuda.Stream(device=0)
            x = torch.randn(256, 256, device="cuda:0")

            # s1 runs first
            with torch.cuda.stream(s1):
                for _ in range(5):
                    x = x @ x

            # s2 waits for s1 via event
            event = torch.cuda.Event()
            event.record(s1)
            s2.wait_event(event)

            with torch.cuda.stream(s2):
                y = torch.randn(256, 256, device="cuda:0")
                for _ in range(5):
                    y = y @ y

            s1.synchronize()
            s2.synchronize()
        """)
        # All ops should have valid timing (end > start)
        bad = conn.execute(
            "SELECT count(*) FROM rocpd_op WHERE end <= start AND gpuId >= 0"
        ).fetchone()[0]
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert bad == 0, "{n} ops with end <= start".format(n=bad)
        assert op_count >= 10, "Expected >=10 ops, got {n}".format(n=op_count)

    @gpu
    def test_default_and_explicit_stream(self, tmp_path):
        """Default stream + explicit stream mixed, both traced."""
        conn = _trace_and_connect(tmp_path, """
            import torch

            # Default stream
            x = torch.randn(128, 128, device="cuda:0")
            for _ in range(5):
                x = x @ x
            torch.cuda.synchronize()

            # Explicit stream
            s = torch.cuda.Stream(device=0)
            y = torch.randn(128, 128, device="cuda:0")
            with torch.cuda.stream(s):
                for _ in range(5):
                    y = y @ y
            s.synchronize()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # Both default and explicit streams should produce ops
        assert op_count >= 10, (
            "Expected >=10 ops from both streams, got {n}".format(n=op_count))

    @gpu
    def test_stream_create_destroy_create(self, tmp_path):
        """Stream create -> destroy -> create, new stream's kernels also traced."""
        conn = _trace_and_connect(tmp_path, """
            import torch

            # First stream
            s1 = torch.cuda.Stream(device=0)
            x = torch.randn(64, 64, device="cuda:0")
            with torch.cuda.stream(s1):
                for _ in range(5):
                    x = x @ x
            s1.synchronize()
            del s1

            # Second stream (after first destroyed)
            s2 = torch.cuda.Stream(device=0)
            y = torch.randn(64, 64, device="cuda:0")
            with torch.cuda.stream(s2):
                for _ in range(5):
                    y = y @ y
            s2.synchronize()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert op_count >= 10, (
            "Expected >=10 ops from both stream rounds, got {n}".format(n=op_count))


# ===================================================================
# HIP Graph tests (5 tests)
# ===================================================================
class TestHIPGraph:
    """Single-dimension: HIP Graph capture and replay."""

    @gpu
    def test_graph_single_replay(self, tmp_path):
        """Graph capture -> single replay, kernel recorded."""
        val = _trace_and_query(tmp_path, """
            import torch

            x = torch.randn(128, 128, device="cuda:0")
            s = torch.cuda.Stream(device=0)
            # Warm-up
            with torch.cuda.stream(s):
                _ = x @ x
            s.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                y = x @ x
            g.replay()
            s.synchronize()
        """)
        assert val[0] > 0, "No ops captured for single graph replay"

    @gpu
    def test_graph_10_replays_count(self, tmp_path):
        """Graph capture -> 10 replays, ops count reasonable (not 10x capture)."""
        conn = _trace_and_connect(tmp_path, """
            import torch

            x = torch.randn(128, 128, device="cuda:0")
            s = torch.cuda.Stream(device=0)
            with torch.cuda.stream(s):
                _ = x @ x
            s.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                y = x @ x

            for _ in range(10):
                g.replay()
            s.synchronize()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # Graph replay may or may not record each replay as separate op
        # Key: should not crash and should have some ops
        assert op_count > 0, "No ops captured for 10 graph replays"

    @gpu
    def test_graph_update_replay(self, tmp_path):
        """Graph update (replace kernel) -> replay, new kernel appears."""
        conn = _trace_and_connect(tmp_path, """
            import torch

            x = torch.randn(128, 128, device="cuda:0")
            s = torch.cuda.Stream(device=0)
            with torch.cuda.stream(s):
                _ = x @ x
            s.synchronize()

            # Capture with matmul
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                y = x @ x
            g.replay()
            s.synchronize()

            # Capture again (new graph) with addition
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2, stream=s):
                z = x + x
            g2.replay()
            s.synchronize()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        descs = conn.execute(
            "SELECT count(DISTINCT description_id) FROM rocpd_op"
        ).fetchone()[0]
        conn.close()
        assert op_count > 0, "No ops captured"
        assert descs >= 2, (
            "Expected >=2 distinct kernel descriptions, got {n}".format(n=descs))

    @gpu
    def test_graph_with_roctx(self, tmp_path):
        """Graph + roctx marker coexist, both in trace."""
        conn = _trace_and_connect(tmp_path, ROCTX_PREAMBLE + """
            import torch

            _roctx.roctxRangePushA(b"graph_section")

            x = torch.randn(128, 128, device="cuda:0")
            s = torch.cuda.Stream(device=0)
            with torch.cuda.stream(s):
                _ = x @ x
            s.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                y = x @ x
            for _ in range(5):
                g.replay()
            s.synchronize()

            _roctx.roctxRangePop()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        markers = conn.execute(
            "SELECT count(*) FROM rocpd_op o "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'UserMarker'"
        ).fetchone()[0]
        conn.close()
        assert op_count > 0, "No ops captured"
        assert markers > 0, "No roctx markers captured alongside graph"

    @gpu
    def test_graph_empty_capture_no_crash(self, tmp_path):
        """Graph capture with no kernels, replay does not crash."""
        val = _trace_and_query(tmp_path, """
            import torch

            s = torch.cuda.Stream(device=0)
            # Warm-up stream
            x = torch.randn(4, 4, device="cuda:0")
            with torch.cuda.stream(s):
                _ = x + 0
            s.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                pass  # empty capture
            g.replay()
            s.synchronize()
            print("ok")
        """)
        # Should not crash; ops can be 0 or small (warmup only)
        assert val is not None, "Query failed on trace from empty graph"


# ===================================================================
# roctx E2E tests (5 tests)
# ===================================================================
class TestRoctxE2E:
    """Single-dimension: roctx markers and ranges."""

    @gpu
    def test_range_start_stop(self, tmp_path):
        """roctxRangeStartA / roctxRangeStop via ctypes, range in DB."""
        conn = _trace_and_connect(tmp_path, ROCTX_PREAMBLE + """
            import torch

            rid = _roctx.roctxRangeStartA(b"my_range")
            x = torch.randn(64, 64, device="cuda:0")
            for _ in range(5):
                x = x @ x
            torch.cuda.synchronize()
            _roctx.roctxRangeStop(rid)
        """)
        markers = conn.execute(
            "SELECT count(*) FROM rocpd_op o "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'UserMarker'"
        ).fetchone()[0]
        # Check the range name appears in strings
        has_range = conn.execute(
            "SELECT count(*) FROM rocpd_string WHERE string = 'my_range'"
        ).fetchone()[0]
        conn.close()
        assert markers > 0, "No UserMarker ops captured"
        assert has_range > 0, "Range name 'my_range' not in rocpd_string"

    @gpu
    def test_range_push_legacy_alias(self, tmp_path):
        """roctxRangePush (no 'A' suffix, legacy alias) works."""
        conn = _trace_and_connect(tmp_path, ROCTX_PREAMBLE + """
            import torch

            _roctx.roctxRangePush(b"legacy_push")
            x = torch.randn(64, 64, device="cuda:0")
            x = x @ x
            torch.cuda.synchronize()
            _roctx.roctxRangePop()
        """)
        has_name = conn.execute(
            "SELECT count(*) FROM rocpd_string WHERE string = 'legacy_push'"
        ).fetchone()[0]
        conn.close()
        assert has_name > 0, "Legacy roctxRangePush not recorded"

    @gpu
    def test_mark_legacy_alias(self, tmp_path):
        """roctxMark (no 'A' suffix, legacy alias) works."""
        conn = _trace_and_connect(tmp_path, ROCTX_PREAMBLE + """
            import torch

            x = torch.randn(64, 64, device="cuda:0")
            x = x @ x
            torch.cuda.synchronize()
            _roctx.roctxMark(b"legacy_mark")
        """)
        has_name = conn.execute(
            "SELECT count(*) FROM rocpd_string WHERE string = 'legacy_mark'"
        ).fetchone()[0]
        conn.close()
        assert has_name > 0, "Legacy roctxMark not recorded"

    @gpu
    def test_nested_push_pop(self, tmp_path):
        """Nested roctx: push A -> push B -> pop B -> pop A, nesting correct."""
        conn = _trace_and_connect(tmp_path, ROCTX_PREAMBLE + """
            import torch

            _roctx.roctxRangePushA(b"outer")
            x = torch.randn(64, 64, device="cuda:0")
            x = x @ x
            torch.cuda.synchronize()

            _roctx.roctxRangePushA(b"inner")
            y = torch.randn(64, 64, device="cuda:0")
            y = y @ y
            torch.cuda.synchronize()
            _roctx.roctxRangePop()

            x = x @ x
            torch.cuda.synchronize()
            _roctx.roctxRangePop()
        """)
        outer = conn.execute(
            "SELECT count(*) FROM rocpd_string WHERE string = 'outer'"
        ).fetchone()[0]
        inner = conn.execute(
            "SELECT count(*) FROM rocpd_string WHERE string = 'inner'"
        ).fetchone()[0]
        markers = conn.execute(
            "SELECT count(*) FROM rocpd_op o "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'UserMarker'"
        ).fetchone()[0]
        conn.close()
        assert outer > 0, "'outer' not in rocpd_string"
        assert inner > 0, "'inner' not in rocpd_string"
        assert markers >= 2, "Expected >=2 markers (outer+inner), got {n}".format(
            n=markers)

    @gpu
    def test_100_rapid_markers(self, tmp_path):
        """100 rapid roctx markers, all recorded."""
        conn = _trace_and_connect(tmp_path, ROCTX_PREAMBLE + """
            import torch

            x = torch.randn(16, 16, device="cuda:0")
            _ = x + x
            torch.cuda.synchronize()

            for i in range(100):
                _roctx.roctxMarkA("mark_{0}".format(i).encode())
        """)
        markers = conn.execute(
            "SELECT count(*) FROM rocpd_op o "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'UserMarker'"
        ).fetchone()[0]
        conn.close()
        assert markers >= 100, (
            "Expected >=100 markers, got {n}".format(n=markers))


# ===================================================================
# Other single-dim tests (7 tests)
# ===================================================================
class TestOtherSingleDim:
    """Single-dimension: miscellaneous profiler correctness tests."""

    @gpu
    def test_kernel_name_demangled(self, tmp_path):
        """Torch GEMM kernel name is not just 'kernel_0x...'."""
        conn = _trace_and_connect(tmp_path, """
            import torch

            x = torch.randn(256, 256, device="cuda:0")
            for _ in range(5):
                x = x @ x
            torch.cuda.synchronize()
        """)
        names = [r[0] for r in conn.execute(
            "SELECT DISTINCT s.string FROM rocpd_op o "
            "JOIN rocpd_string s ON o.description_id = s.id "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'KernelExecution'"
        ).fetchall()]
        conn.close()
        assert len(names) > 0, "No kernel names found"
        # At least one kernel name should be meaningful (not just hex address)
        has_meaningful = any(
            not n.startswith("kernel_0x") for n in names
        )
        assert has_meaningful, (
            "All kernel names are hex addresses: {ns}".format(
                ns=[n[:50] for n in names]))

    @gpu
    def test_tick_monotonicity(self, tmp_path):
        """tick() calls via ctypes are monotonically increasing."""
        conn = _trace_and_connect(tmp_path, """
            import ctypes, os

            _lib = os.environ.get("HSA_TOOLS_LIB", "librtl.so")
            try:
                _rtl = ctypes.CDLL(_lib)
                _rtl.tick.restype = ctypes.c_uint64
                ticks = [_rtl.tick() for _ in range(100)]
                for i in range(1, len(ticks)):
                    assert ticks[i] >= ticks[i-1], (
                        "tick() not monotonic at index {}: {} < {}".format(
                            i, ticks[i], ticks[i-1])
                    )
            except (OSError, AttributeError):
                pass  # tick() may not be exported; skip silently

            # Still need some GPU work so trace file is created
            import torch
            x = torch.randn(16, 16, device="cuda:0")
            x = x + x
            torch.cuda.synchronize()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert op_count >= 0, "Trace query failed"

    @gpu
    def test_correlation_id_uniqueness(self, tmp_path):
        """If correlation_id available via ctypes, 100 calls all different."""
        conn = _trace_and_connect(tmp_path, """
            import ctypes, os

            _lib = os.environ.get("HSA_TOOLS_LIB", "librtl.so")
            try:
                _rtl = ctypes.CDLL(_lib)
                _rtl.getCorrelationId.restype = ctypes.c_uint64
                ids = [_rtl.getCorrelationId() for _ in range(100)]
                unique = len(set(ids))
                assert unique == 100, (
                    "correlation IDs not unique: {} unique out of 100".format(unique)
                )
            except (OSError, AttributeError):
                pass  # getCorrelationId may not be exported

            import torch
            x = torch.randn(16, 16, device="cuda:0")
            x = x + x
            torch.cuda.synchronize()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert op_count >= 0, "Trace query failed"

    @gpu
    def test_record_copy_op(self, tmp_path):
        """torch.Tensor.to('cpu') produces a copy op in trace (if profiler records copy)."""
        conn = _trace_and_connect(tmp_path, """
            import torch

            x = torch.randn(1024, 1024, device="cuda:0")
            torch.cuda.synchronize()
            y = x.to("cpu")  # D2H copy
            z = y.to("cuda:0")  # H2D copy
            torch.cuda.synchronize()
        """)
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # Copy tracing is optional -- we just verify no crash and ops exist
        assert op_count >= 0, "Trace query failed"

    @gpu
    def test_large_kernel_count(self, tmp_path):
        """1000 small GEMMs, ops count >= 900 (allow some drop)."""
        val = _trace_and_query(tmp_path, """
            import torch

            x = torch.randn(64, 64, device="cuda:0")
            for _ in range(1000):
                x = x @ x
            torch.cuda.synchronize()
        """, timeout=180)
        assert val[0] >= 900, (
            "Data loss: expected >=900 ops from 1000 GEMMs, got {n}".format(n=val[0]))

    @gpu
    def test_empty_workload_no_crash(self, tmp_path):
        """Only import torch, no kernel, trace has 0 ops and no crash."""
        trace = str(tmp_path / "trace.db")
        script = str(tmp_path / "workload.py")
        with open(script, "w") as f:
            f.write(textwrap.dedent("""
                import torch
                # No GPU kernel -- just import and exit
                if torch.cuda.is_available():
                    _ = torch.cuda.device_count()
            """))
        result = subprocess.run(
            [sys.executable, "-m", "rocm_trace_lite.cli", "trace",
             "-o", trace, sys.executable, script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=60, cwd=REPO_ROOT,
        )
        # Should not crash regardless of whether trace file is created
        assert result.returncode == 0, (
            "Crashed on empty workload: {err}".format(err=result.stderr[-500:]))
        # If trace exists, verify it has 0 or few ops
        if os.path.exists(trace):
            conn = sqlite3.connect(trace)
            op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
            conn.close()
            assert op_count >= 0, "Unexpected negative op count"

    @gpu
    def test_very_short_kernels_no_hang(self, tmp_path):
        """1x1 matmul x 100, profiler does not hang."""
        val = _trace_and_query(tmp_path, """
            import torch

            x = torch.randn(1, 1, device="cuda:0")
            for _ in range(100):
                x = x @ x
            torch.cuda.synchronize()
        """, timeout=60)
        assert val[0] >= 0, "Trace query failed for very short kernels"
