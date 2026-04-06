"""GPU integration tests for the rtl CLI with multi-stream workloads.

These tests require a ROCm GPU. Skip gracefully if unavailable.
Run with: pytest tests/test_cli_multistream.py -v

Ref: Issue #22
"""
import json
import os
import sqlite3
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _has_gpu():
    """Check if ROCm GPU is available."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch; assert torch.cuda.is_available()"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        return r.returncode == 0
    except Exception:
        return False


def _gpu_count():
    """Return the number of visible GPUs."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        if r.returncode == 0:
            return int(r.stdout.decode().strip())
    except Exception:
        pass
    return 0


def _skip_if_no_gpu():
    if not _has_gpu():
        pytest.skip("No ROCm GPU available")


def _skip_if_less_than_2_gpus():
    _skip_if_no_gpu()
    if _gpu_count() < 2:
        pytest.skip("Need >=2 GPUs")


def _run_rtl(*args, timeout=120):
    """Run rtl CLI via python -m."""
    r = subprocess.run(
        [sys.executable, "-m", "rocm_trace_lite.cli"] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, cwd=REPO_ROOT,
    )
    return r.returncode, r.stdout.decode(), r.stderr.decode()


def _write_script(tmp_path, name, code):
    """Write a Python script to tmp_path and return its path."""
    script = tmp_path / name
    script.write_text(textwrap.dedent(code))
    return str(script)


class TestRtl4StreamsSingleGPU:
    """rtl trace with 4 concurrent streams on a single GPU."""

    def test_rtl_4_streams_single_gpu(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.rpd")
        script = _write_script(tmp_path, "workload.py", """\
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
        """)
        rc, out, err = _run_rtl("trace", "-o", trace, sys.executable, script)
        assert rc == 0, f"rtl trace failed: {err[-500:]}"
        assert os.path.exists(trace), "No trace file created"

        conn = sqlite3.connect(trace)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # 4 streams x 50 matmuls = 200 GEMM ops minimum
        assert count >= 200, f"Expected >=200 ops, got {count}"


class TestRtlMultiGPU:
    """rtl trace across multiple GPUs."""

    def test_rtl_multi_gpu(self, tmp_path):
        _skip_if_less_than_2_gpus()
        trace = str(tmp_path / "multi_gpu.rpd")
        script = _write_script(tmp_path, "workload.py", """\
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
        """)
        rc, out, err = _run_rtl("trace", "-o", trace, sys.executable, script)
        assert rc == 0, f"rtl trace failed: {err[-500:]}"
        assert os.path.exists(trace), "No trace file created"

        conn = sqlite3.connect(trace)
        gpu_ops = dict(conn.execute(
            "SELECT gpuId, count(*) FROM rocpd_op GROUP BY gpuId"
        ).fetchall())
        conn.close()
        assert 0 in gpu_ops, "No ops on GPU 0"
        assert 1 in gpu_ops, "No ops on GPU 1"
        assert gpu_ops[0] >= 30, f"GPU 0: expected >=30 ops, got {gpu_ops[0]}"
        assert gpu_ops[1] >= 30, f"GPU 1: expected >=30 ops, got {gpu_ops[1]}"


class TestRtlGraphPlusStreams:
    """rtl trace with HIP graph replay + eager dispatch on separate stream."""

    def test_rtl_graph_plus_streams(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.rpd")
        script = _write_script(tmp_path, "workload.py", """\
            import torch

            # --- Graph capture and replay on stream 0 ---
            stream_graph = torch.cuda.Stream(device=0)
            x = torch.randn(128, 128, device="cuda:0")

            # Warm-up for graph capture
            with torch.cuda.stream(stream_graph):
                for _ in range(3):
                    x = x @ x
            stream_graph.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream_graph):
                with torch.cuda.graph(g):
                    x = x @ x

            # Replay graph 20 times
            for _ in range(20):
                g.replay()
            stream_graph.synchronize()

            # --- Eager dispatch on a separate stream ---
            stream_eager = torch.cuda.Stream(device=0)
            y = torch.randn(128, 128, device="cuda:0")
            with torch.cuda.stream(stream_eager):
                for _ in range(20):
                    y = y @ y
            stream_eager.synchronize()

            torch.cuda.synchronize()
        """)
        rc, out, err = _run_rtl("trace", "-o", trace, sys.executable, script)
        assert rc == 0, f"rtl trace crashed: {err[-500:]}"
        assert os.path.exists(trace), "No trace file created"

        conn = sqlite3.connect(trace)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # Graph replays + eager ops should produce some trace entries
        assert count > 0, "Trace has no ops"


class TestRtlSummaryOnMultistream:
    """rtl summary on a multi-stream trace."""

    def test_rtl_summary_on_multistream(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.rpd")
        script = _write_script(tmp_path, "workload.py", """\
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
        """)

        # Step 1: Collect trace
        rc, out, err = _run_rtl("trace", "-o", trace, sys.executable, script)
        assert rc == 0, f"rtl trace failed: {err[-500:]}"
        assert os.path.exists(trace)

        # Step 2: Run summary
        rc, out, err = _run_rtl("summary", trace)
        assert rc == 0, f"rtl summary failed: {err[-500:]}"

        # Assert output contains kernel names (GEMM kernels contain "Cijk" or "gemm")
        out_lower = out.lower()
        has_kernel = "cijk" in out_lower or "gemm" in out_lower or "matmul" in out_lower
        assert has_kernel, f"Summary output missing kernel names:\n{out[:1000]}"

        # Assert output contains GPU utilization info
        assert "GPU" in out, f"Summary output missing GPU utilization info:\n{out[:1000]}"


class TestRtlConvertMultistream:
    """rtl convert on a multi-stream trace produces valid JSON."""

    def test_rtl_convert_multistream(self, tmp_path):
        _skip_if_no_gpu()
        trace = str(tmp_path / "trace.rpd")
        json_out = str(tmp_path / "trace.json")
        script = _write_script(tmp_path, "workload.py", """\
            import torch, threading

            def worker(stream, n):
                x = torch.randn(256, 256, device="cuda:0")
                with torch.cuda.stream(stream):
                    for _ in range(n):
                        x = x @ x
                stream.synchronize()

            N_STREAMS, N_ITERS = 2, 20
            streams = [torch.cuda.Stream(device=0) for _ in range(N_STREAMS)]
            threads = [threading.Thread(target=worker, args=(s, N_ITERS)) for s in streams]
            for t in threads: t.start()
            for t in threads: t.join()
            torch.cuda.synchronize()
        """)

        # Step 1: Collect trace
        rc, out, err = _run_rtl("trace", "-o", trace, sys.executable, script)
        assert rc == 0, f"rtl trace failed: {err[-500:]}"
        assert os.path.exists(trace)

        # Step 2: Convert to JSON
        rc, out, err = _run_rtl("convert", trace, "-o", json_out)
        assert rc == 0, f"rtl convert failed: {err[-500:]}"
        assert os.path.exists(json_out), "JSON output file not created"
        assert os.path.getsize(json_out) > 0, "JSON output file is empty"

        # Step 3: Validate JSON structure
        with open(json_out) as f:
            data = json.load(f)
        # Perfetto JSON format uses traceEvents list
        if isinstance(data, dict):
            assert "traceEvents" in data, f"Missing traceEvents key, got keys: {list(data.keys())}"
            events = data["traceEvents"]
        elif isinstance(data, list):
            events = data
        else:
            raise AssertionError(f"Unexpected JSON type: {type(data)}")
        assert len(events) > 0, "JSON trace has no events"
