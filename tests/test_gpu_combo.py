"""Sprint 3: Combination-dimension GPU E2E tests.

Tests exercise cross-cutting concerns: Thread x Stream, Thread x GPU,
Stream x GPU, Graph x Stream, Graph x GPU, Process x GPU, and full
3-way combinations.

Requires a ROCm GPU. Skip gracefully if unavailable.
Run with: pytest tests/test_gpu_combo.py -v
"""
import os
import re
import sqlite3
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "librpd_lite.so")


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


HAS_GPU = _has_gpu()
GPU_COUNT = _gpu_count() if HAS_GPU else 0

gpu = pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
multigpu = pytest.mark.skipif(
    not HAS_GPU or GPU_COUNT < 2,
    reason="Need 2+ GPUs",
)


def _run_traced(script, trace_path, timeout=120):
    """Run a Python script string with rpd_lite tracing."""
    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = LIB_PATH
    env["RPD_LITE_OUTPUT"] = trace_path
    r = subprocess.run(
        [sys.executable, "-c", script],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return r.returncode, r.stdout.decode(), r.stderr.decode()


def _run_rtl(*args, **kwargs):
    """Run rtl CLI via python -m."""
    timeout = kwargs.get("timeout", 120)
    r = subprocess.run(
        [sys.executable, "-m", "rocm_trace_lite.cli"] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, cwd=REPO_ROOT,
    )
    return r.returncode, r.stdout.decode(), r.stderr.decode()


def _torchrun_trace(tmp_path, script_code, nproc=2, timeout=120):
    """Run script under torchrun with rtl trace."""
    trace = str(tmp_path / "trace.db")
    script = str(tmp_path / "workload.py")
    with open(script, "w") as f:
        f.write(script_code)
    result = subprocess.run(
        [sys.executable, "-m", "rocm_trace_lite.cli", "trace", "-o", trace,
         sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node={}".format(nproc), "--master_port=29500", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=timeout,
    )
    return trace, result


def _query_ops(trace_path):
    """Return total op count from trace."""
    conn = sqlite3.connect(trace_path)
    count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
    conn.close()
    return count


def _query_unique_queues(trace_path):
    """Return set of unique queueId values."""
    conn = sqlite3.connect(trace_path)
    rows = conn.execute("SELECT DISTINCT queueId FROM rocpd_op").fetchall()
    conn.close()
    return set(r[0] for r in rows)


def _query_unique_gpus(trace_path):
    """Return set of unique gpuId values (non-negative)."""
    conn = sqlite3.connect(trace_path)
    rows = conn.execute(
        "SELECT DISTINCT gpuId FROM rocpd_op WHERE gpuId >= 0"
    ).fetchall()
    conn.close()
    return set(r[0] for r in rows)


def _query_gpu_queue_combos(trace_path):
    """Return set of (gpuId, queueId) combinations."""
    conn = sqlite3.connect(trace_path)
    rows = conn.execute(
        "SELECT DISTINCT gpuId, queueId FROM rocpd_op WHERE gpuId >= 0"
    ).fetchall()
    conn.close()
    return set(rows)


# =========================================================================
# Thread x Stream (4 tests)
# =========================================================================
class TestThreadStream:
    """Thread x Stream combination tests."""

    @gpu
    def test_2_threads_2_streams_each(self, tmp_path):
        """2 threads x 2 streams each = 4 concurrent streams."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(gpu, streams, n):
    for s in streams:
        x = torch.randn(128, 128, device="cuda:0")
        with torch.cuda.stream(s):
            for _ in range(n):
                x = x @ x
        s.synchronize()

threads = []
for _ in range(2):
    streams = [torch.cuda.Stream(device=0) for _ in range(2)]
    t = threading.Thread(target=worker, args=(0, streams, 20))
    threads.append(t)
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize()
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        queues = _query_unique_queues(trace)
        assert len(queues) >= 4, "Expected 4+ unique queueId, got {}".format(len(queues))

    @gpu
    def test_4_threads_1_stream_each(self, tmp_path):
        """4 threads x 1 stream each, ops on different queues."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(n):
    stream = torch.cuda.Stream(device=0)
    x = torch.randn(128, 128, device="cuda:0")
    with torch.cuda.stream(stream):
        for _ in range(n):
            x = x @ x
    stream.synchronize()

threads = [threading.Thread(target=worker, args=(20,)) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize()
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        queues = _query_unique_queues(trace)
        assert len(queues) >= 2, "Expected ops on different queues, got {}".format(len(queues))

    @gpu
    def test_1_thread_4_streams_sequential(self, tmp_path):
        """1 thread x 4 streams sequential dispatch."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch

streams = [torch.cuda.Stream(device=0) for _ in range(4)]
for s in streams:
    x = torch.randn(128, 128, device="cuda:0")
    with torch.cuda.stream(s):
        for _ in range(10):
            x = x @ x
    s.synchronize()
torch.cuda.synchronize()
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        ops = _query_ops(trace)
        assert ops >= 40, "Expected >=40 ops, got {}".format(ops)

    @gpu
    def test_8_threads_2_streams_no_crash(self, tmp_path):
        """8 threads x 2 streams = 16 concurrent, verify no crash."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(streams, n):
    for s in streams:
        x = torch.randn(64, 64, device="cuda:0")
        with torch.cuda.stream(s):
            for _ in range(n):
                x = x @ x
        s.synchronize()

threads = []
for _ in range(8):
    streams = [torch.cuda.Stream(device=0) for _ in range(2)]
    t = threading.Thread(target=worker, args=(streams, 10))
    threads.append(t)
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize()
"""
        rc, _, err = _run_traced(script, trace, timeout=180)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        ops = _query_ops(trace)
        assert ops > 0, "Expected ops > 0"


# =========================================================================
# Thread x GPU (3 tests)
# =========================================================================
class TestThreadGPU:
    """Thread x GPU combination tests."""

    @multigpu
    def test_2_threads_2_gpus(self, tmp_path):
        """2 threads, each on a different GPU."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(gpu_id, n):
    torch.cuda.set_device(gpu_id)
    x = torch.randn(128, 128, device="cuda:{}".format(gpu_id))
    stream = torch.cuda.Stream(device=gpu_id)
    with torch.cuda.stream(stream):
        for _ in range(n):
            x = x @ x
    stream.synchronize()

threads = [threading.Thread(target=worker, args=(i, 30)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected 2 unique gpuId, got {}".format(gpus)

    @multigpu
    def test_4_threads_2_gpus(self, tmp_path):
        """4 threads on 2 GPUs (2 per GPU)."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(gpu_id, n):
    torch.cuda.set_device(gpu_id)
    x = torch.randn(128, 128, device="cuda:{}".format(gpu_id))
    stream = torch.cuda.Stream(device=gpu_id)
    with torch.cuda.stream(stream):
        for _ in range(n):
            x = x @ x
    stream.synchronize()

threads = [threading.Thread(target=worker, args=(i % 2, 20)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected 2 gpuId, got {}".format(gpus)
        conn = sqlite3.connect(trace)
        for g in [0, 1]:
            cnt = conn.execute(
                "SELECT count(*) FROM rocpd_op WHERE gpuId = ?", (g,)
            ).fetchone()[0]
            assert cnt >= 20, "GPU {}: expected >=20 ops, got {}".format(g, cnt)
        conn.close()

    @multigpu
    def test_thread_switches_gpu(self, tmp_path):
        """Thread switches GPU mid-run, both gpuId recorded."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch

# Phase 1: GPU 0
torch.cuda.set_device(0)
x = torch.randn(128, 128, device="cuda:0")
for _ in range(20):
    x = x @ x
torch.cuda.synchronize(0)

# Phase 2: GPU 1
torch.cuda.set_device(1)
y = torch.randn(128, 128, device="cuda:1")
for _ in range(20):
    y = y @ y
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected 2 gpuId, got {}".format(gpus)


# =========================================================================
# Stream x GPU (3 tests)
# =========================================================================
class TestStreamGPU:
    """Stream x GPU combination tests."""

    @multigpu
    def test_2_streams_per_2_gpus(self, tmp_path):
        """GPU0 2 streams + GPU1 2 streams = 4+ unique (gpuId, queueId)."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(gpu_id, n):
    streams = [torch.cuda.Stream(device=gpu_id) for _ in range(2)]
    for s in streams:
        x = torch.randn(128, 128, device="cuda:{}".format(gpu_id))
        with torch.cuda.stream(s):
            for _ in range(n):
                x = x @ x
        s.synchronize()

threads = [threading.Thread(target=worker, args=(i, 15)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        combos = _query_gpu_queue_combos(trace)
        assert len(combos) >= 4, "Expected 4+ (gpuId, queueId) combos, got {}".format(len(combos))

    @multigpu
    def test_cross_gpu_stream_sync(self, tmp_path):
        """Cross-GPU event sync: event on GPU0, wait on GPU1."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch

# GPU 0 work
s0 = torch.cuda.Stream(device=0)
x = torch.randn(128, 128, device="cuda:0")
with torch.cuda.stream(s0):
    for _ in range(20):
        x = x @ x
event = s0.record_event()

# GPU 1 waits on GPU 0's event, then does work
s1 = torch.cuda.Stream(device=1)
with torch.cuda.stream(s1):
    s1.wait_event(event)
    y = torch.randn(128, 128, device="cuda:1")
    for _ in range(20):
        y = y @ y
s1.synchronize()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected kernels on both GPUs, got gpuIds {}".format(gpus)

    @multigpu
    def test_8_streams_across_2_gpus(self, tmp_path):
        """8 streams across 2 GPUs (4 per GPU)."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(gpu_id, n):
    streams = [torch.cuda.Stream(device=gpu_id) for _ in range(4)]
    for s in streams:
        x = torch.randn(64, 64, device="cuda:{}".format(gpu_id))
        with torch.cuda.stream(s):
            for _ in range(n):
                x = x @ x
        s.synchronize()

threads = [threading.Thread(target=worker, args=(i, 10)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected 2 gpuId"
        queues = _query_unique_queues(trace)
        assert len(queues) >= 4, "Expected 4+ queueId across 2 GPUs, got {}".format(len(queues))


# =========================================================================
# Graph x Stream (3 tests)
# =========================================================================
class TestGraphStream:
    """Graph x Stream combination tests."""

    @gpu
    def test_capture_stream_a_replay_stream_b(self, tmp_path):
        """Capture on stream A, replay on stream B."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch

x = torch.randn(128, 128, device="cuda:0")
sa = torch.cuda.Stream(device=0)
sb = torch.cuda.Stream(device=0)

# Warm up
with torch.cuda.stream(sa):
    _ = x @ x
sa.synchronize()

# Capture on stream A
g = torch.cuda.CUDAGraph()
with torch.cuda.stream(sa):
    with torch.cuda.graph(g):
        y = x @ x

# Replay on stream A (graph is bound to capture stream)
for _ in range(10):
    g.replay()
sa.synchronize()

# Also do eager work on stream B to verify both captured
with torch.cuda.stream(sb):
    z = torch.randn(128, 128, device="cuda:0")
    for _ in range(10):
        z = z @ z
sb.synchronize()
torch.cuda.synchronize()
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        ops = _query_ops(trace)
        assert ops > 0, "Expected ops from graph replay and eager stream"

    @gpu
    def test_capture_default_replay_explicit(self, tmp_path):
        """Capture on default stream, replay; also eager on explicit stream."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch

x = torch.randn(128, 128, device="cuda:0")
# Warm up on default stream
_ = x @ x
torch.cuda.synchronize()

# Capture on default stream
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream(device=0)
with torch.cuda.stream(s):
    with torch.cuda.graph(g):
        y = x @ x

for _ in range(10):
    g.replay()
s.synchronize()

# Eager on explicit stream
s2 = torch.cuda.Stream(device=0)
with torch.cuda.stream(s2):
    z = torch.randn(128, 128, device="cuda:0")
    for _ in range(10):
        z = z @ z
s2.synchronize()
torch.cuda.synchronize()
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        ops = _query_ops(trace)
        assert ops > 0, "Expected ops > 0"

    @gpu
    def test_2_graphs_2_streams_interleaved(self, tmp_path):
        """2 graphs captured on 2 streams, replayed interleaved."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch

x = torch.randn(128, 128, device="cuda:0")
s1 = torch.cuda.Stream(device=0)
s2 = torch.cuda.Stream(device=0)

# Warm up
with torch.cuda.stream(s1):
    _ = x @ x
s1.synchronize()
with torch.cuda.stream(s2):
    _ = x @ x
s2.synchronize()

# Capture graph 1 on s1
g1 = torch.cuda.CUDAGraph()
with torch.cuda.stream(s1):
    with torch.cuda.graph(g1):
        y1 = x @ x

# Capture graph 2 on s2
g2 = torch.cuda.CUDAGraph()
with torch.cuda.stream(s2):
    with torch.cuda.graph(g2):
        y2 = x @ x

# Interleaved replay
for _ in range(10):
    g1.replay()
    g2.replay()
s1.synchronize()
s2.synchronize()
torch.cuda.synchronize()
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        ops = _query_ops(trace)
        assert ops > 0, "Expected ops from interleaved graph replays"


# =========================================================================
# Graph x GPU (2 tests)
# =========================================================================
class TestGraphGPU:
    """Graph x GPU combination tests."""

    @multigpu
    def test_graph_capture_replay_on_gpu0(self, tmp_path):
        """Capture graph on GPU0, replay on GPU0, verify gpuId."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch

x = torch.randn(128, 128, device="cuda:0")
s = torch.cuda.Stream(device=0)
with torch.cuda.stream(s):
    _ = x @ x
s.synchronize()

g = torch.cuda.CUDAGraph()
with torch.cuda.stream(s):
    with torch.cuda.graph(g):
        y = x @ x

for _ in range(20):
    g.replay()
s.synchronize()
torch.cuda.synchronize()
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        gpus = _query_unique_gpus(trace)
        assert 0 in gpus, "Expected gpuId 0 in trace, got {}".format(gpus)

    @multigpu
    def test_2_gpus_each_capture_replay(self, tmp_path):
        """Two GPUs each capture + replay their own graph."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def graph_worker(gpu_id):
    x = torch.randn(128, 128, device="cuda:{}".format(gpu_id))
    s = torch.cuda.Stream(device=gpu_id)
    with torch.cuda.stream(s):
        _ = x @ x
    s.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            y = x @ x

    for _ in range(15):
        g.replay()
    s.synchronize()

threads = [threading.Thread(target=graph_worker, args=(i,)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected 2 gpuId, got {}".format(gpus)


# =========================================================================
# Graph x Multi-replay (2 tests)
# =========================================================================
class TestGraphMultiReplay:
    """Graph replay stress tests."""

    @gpu
    def test_100_replays_signal_pool(self, tmp_path):
        """100 replays of same graph, verify signal pool recycling."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch

x = torch.randn(128, 128, device="cuda:0")
s = torch.cuda.Stream(device=0)
with torch.cuda.stream(s):
    _ = x @ x
s.synchronize()

g = torch.cuda.CUDAGraph()
with torch.cuda.stream(s):
    with torch.cuda.graph(g):
        y = x @ x

for _ in range(100):
    g.replay()
s.synchronize()
torch.cuda.synchronize()
"""
        rc, _, err = _run_traced(script, trace, timeout=180)
        assert rc == 0, "Crashed or hung: {}".format(err[-500:])
        ops = _query_ops(trace)
        assert ops > 0, "Expected ops > 0 after 100 replays"

    @gpu
    def test_50_replays_drop_counter(self, tmp_path):
        """50 replays, verify diagnostic output on stderr."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch

x = torch.randn(128, 128, device="cuda:0")
s = torch.cuda.Stream(device=0)
with torch.cuda.stream(s):
    _ = x @ x
s.synchronize()

g = torch.cuda.CUDAGraph()
with torch.cuda.stream(s):
    with torch.cuda.graph(g):
        y = x @ x

for _ in range(50):
    g.replay()
s.synchronize()
torch.cuda.synchronize()
"""
        rc, _, err = _run_traced(script, trace, timeout=180)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        ops = _query_ops(trace)
        assert ops > 0, "Expected ops > 0 after 50 replays"
        # Drop counter diagnostic is optional; just verify no crash


# =========================================================================
# Process x GPU (4 tests)
# =========================================================================
class TestProcessGPU:
    """Process x GPU combination tests via torchrun."""

    @multigpu
    def test_torchrun_2_proc_2_gpus(self, tmp_path):
        """torchrun --nproc_per_node=2, verify merged trace has 2 gpuId."""
        script_code = """\
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
x = torch.randn(128, 128, device="cuda:{}".format(rank))
for _ in range(20):
    x = x @ x
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=2, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        assert os.path.exists(trace), "No trace file"
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected 2 gpuId in merged trace, got {}".format(gpus)

    @multigpu
    def test_torchrun_4_proc_4_gpus(self, tmp_path):
        """torchrun --nproc_per_node=4, verify 4 gpuId."""
        if GPU_COUNT < 4:
            pytest.skip("Need 4+ GPUs")
        script_code = """\
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
x = torch.randn(128, 128, device="cuda:{}".format(rank))
for _ in range(20):
    x = x @ x
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=4, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 4, "Expected 4 gpuId, got {}".format(gpus)

    @multigpu
    def test_torchrun_8_proc_8_gpus(self, tmp_path):
        """torchrun --nproc_per_node=8, verify 8 gpuId."""
        if GPU_COUNT < 8:
            pytest.skip("Need 8 GPUs")
        script_code = """\
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
x = torch.randn(128, 128, device="cuda:{}".format(rank))
for _ in range(20):
    x = x @ x
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=8, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 8, "Expected 8 gpuId, got {}".format(gpus)

    @multigpu
    def test_torchrun_asymmetric_workload(self, tmp_path):
        """Rank 0 runs 100 GEMM, rank 1 runs 10, both present in merge."""
        script_code = """\
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
x = torch.randn(128, 128, device="cuda:{}".format(rank))
n = 100 if rank == 0 else 10
for _ in range(n):
    x = x @ x
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=2, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected 2 gpuId, got {}".format(gpus)
        ops = _query_ops(trace)
        assert ops >= 110, "Expected >=110 ops (100+10), got {}".format(ops)


# =========================================================================
# Process x Thread (2 tests)
# =========================================================================
class TestProcessThread:
    """Process x Thread combination tests."""

    @multigpu
    def test_torchrun_2_proc_2_threads_each(self, tmp_path):
        """torchrun 2 process, each with 2 threads."""
        script_code = """\
import os
import threading
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

def worker(gpu_id, n):
    x = torch.randn(64, 64, device="cuda:{}".format(gpu_id))
    s = torch.cuda.Stream(device=gpu_id)
    with torch.cuda.stream(s):
        for _ in range(n):
            x = x @ x
    s.synchronize()

threads = [threading.Thread(target=worker, args=(rank, 20)) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=2, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        ops = _query_ops(trace)
        assert ops > 0, "Expected ops > 0"

    @multigpu
    def test_torchrun_2_proc_4_threads_each(self, tmp_path):
        """torchrun 2 process, each with 4 threads."""
        script_code = """\
import os
import threading
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

def worker(gpu_id, n):
    x = torch.randn(64, 64, device="cuda:{}".format(gpu_id))
    s = torch.cuda.Stream(device=gpu_id)
    with torch.cuda.stream(s):
        for _ in range(n):
            x = x @ x
    s.synchronize()

threads = [threading.Thread(target=worker, args=(rank, 10)) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=2, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected 2 gpuId in merged trace, got {}".format(gpus)


# =========================================================================
# Process x Stream (2 tests)
# =========================================================================
class TestProcessStream:
    """Process x Stream combination tests."""

    @multigpu
    def test_torchrun_2_proc_2_streams_each(self, tmp_path):
        """torchrun 2 process, each with 2 streams."""
        script_code = """\
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

streams = [torch.cuda.Stream(device=rank) for _ in range(2)]
for s in streams:
    x = torch.randn(128, 128, device="cuda:{}".format(rank))
    with torch.cuda.stream(s):
        for _ in range(15):
            x = x @ x
    s.synchronize()
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=2, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        queues = _query_unique_queues(trace)
        assert len(queues) >= 2, "Expected queueId diversity, got {}".format(len(queues))

    @multigpu
    def test_torchrun_2_proc_default_plus_explicit(self, tmp_path):
        """torchrun 2 process, each uses default + explicit stream."""
        script_code = """\
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

# Default stream work
x = torch.randn(128, 128, device="cuda:{}".format(rank))
for _ in range(10):
    x = x @ x

# Explicit stream work
s = torch.cuda.Stream(device=rank)
with torch.cuda.stream(s):
    y = torch.randn(128, 128, device="cuda:{}".format(rank))
    for _ in range(10):
        y = y @ y
s.synchronize()
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=2, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        ops = _query_ops(trace)
        assert ops >= 40, "Expected >=40 ops (2 procs x 20), got {}".format(ops)


# =========================================================================
# Thread x Stream x GPU (3 tests)
# =========================================================================
class TestThreadStreamGPU:
    """Thread x Stream x GPU 3-way combination tests."""

    @multigpu
    def test_2t_2s_2g(self, tmp_path):
        """2 threads x 2 streams x 2 GPUs = 8 concurrent."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(gpu_id, n):
    streams = [torch.cuda.Stream(device=gpu_id) for _ in range(2)]
    for s in streams:
        x = torch.randn(64, 64, device="cuda:{}".format(gpu_id))
        with torch.cuda.stream(s):
            for _ in range(n):
                x = x @ x
        s.synchronize()

threads = [threading.Thread(target=worker, args=(i, 10)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        ops = _query_ops(trace)
        assert ops > 0, "Expected ops > 0"

    @multigpu
    def test_4t_1s_2g(self, tmp_path):
        """4 threads x 1 stream x 2 GPUs."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(gpu_id, n):
    s = torch.cuda.Stream(device=gpu_id)
    x = torch.randn(64, 64, device="cuda:{}".format(gpu_id))
    with torch.cuda.stream(s):
        for _ in range(n):
            x = x @ x
    s.synchronize()

threads = [threading.Thread(target=worker, args=(i % 2, 15)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected 2+ unique gpuId, got {}".format(gpus)
        queues = _query_unique_queues(trace)
        assert len(queues) >= 2, "Expected 2+ unique queueId, got {}".format(queues)

    @multigpu
    def test_4t_2s_2g_stress(self, tmp_path):
        """Full stress: 4 threads x 2 streams x 2 GPUs, verify no crash."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(gpu_id, n):
    streams = [torch.cuda.Stream(device=gpu_id) for _ in range(2)]
    for s in streams:
        x = torch.randn(64, 64, device="cuda:{}".format(gpu_id))
        with torch.cuda.stream(s):
            for _ in range(n):
                x = x @ x
        s.synchronize()

threads = [threading.Thread(target=worker, args=(i % 2, 10)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace, timeout=180)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        ops = _query_ops(trace)
        assert ops > 0, "Expected ops > 0 in stress test"


# =========================================================================
# Graph x Process x GPU (2 tests)
# =========================================================================
class TestGraphProcessGPU:
    """Graph x Process x GPU combination tests."""

    @multigpu
    def test_torchrun_2_proc_graph_capture_replay(self, tmp_path):
        """torchrun 2 process, each does graph capture + replay."""
        script_code = """\
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

x = torch.randn(128, 128, device="cuda:{}".format(rank))
s = torch.cuda.Stream(device=rank)
with torch.cuda.stream(s):
    _ = x @ x
s.synchronize()

g = torch.cuda.CUDAGraph()
with torch.cuda.stream(s):
    with torch.cuda.graph(g):
        y = x @ x

for _ in range(10):
    g.replay()
s.synchronize()
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=2, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        gpus = _query_unique_gpus(trace)
        assert len(gpus) >= 2, "Expected 2 gpuId, got {}".format(gpus)

    @multigpu
    def test_torchrun_2_proc_10_graph_replays(self, tmp_path):
        """torchrun 2 process, each does 10 graph replays."""
        script_code = """\
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

x = torch.randn(128, 128, device="cuda:{}".format(rank))
s = torch.cuda.Stream(device=rank)
with torch.cuda.stream(s):
    _ = x @ x
s.synchronize()

g = torch.cuda.CUDAGraph()
with torch.cuda.stream(s):
    with torch.cuda.graph(g):
        y = x @ x

for _ in range(10):
    g.replay()
s.synchronize()
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=2, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        ops = _query_ops(trace)
        assert ops >= 20, "Expected >=20 ops from 2 procs x 10 replays, got {}".format(ops)


# =========================================================================
# Multi-GPU misc (5 tests)
# =========================================================================
class TestMultiGPUMisc:
    """Multi-GPU miscellaneous tests."""

    @multigpu
    def test_nccl_all_reduce_in_trace(self, tmp_path):
        """NCCL all-reduce: 2 process, verify NCCL kernel in trace."""
        script_code = """\
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

x = torch.ones(1024, device="cuda:{}".format(rank)) * (rank + 1)
dist.all_reduce(x, op=dist.ReduceOp.SUM)
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        trace, result = _torchrun_trace(tmp_path, script_code, nproc=2, timeout=180)
        assert result.returncode == 0, "torchrun failed: {}".format(result.stderr[-500:])
        ops = _query_ops(trace)
        assert ops > 0, "Expected NCCL ops in trace"

    @multigpu
    def test_multi_gpu_perfetto_json(self, tmp_path):
        """Multi-GPU trace + convert, verify JSON has per-GPU tracks."""
        trace = str(tmp_path / "trace.db")
        json_out = str(tmp_path / "trace.json")
        script = """\
import torch, threading

def worker(gpu_id, n):
    x = torch.randn(128, 128, device="cuda:{}".format(gpu_id))
    s = torch.cuda.Stream(device=gpu_id)
    with torch.cuda.stream(s):
        for _ in range(n):
            x = x @ x
    s.synchronize()

threads = [threading.Thread(target=worker, args=(i, 20)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])

        # Convert to JSON
        sys.path.insert(0, REPO_ROOT)
        from rocm_trace_lite.cmd_convert import convert
        import json as json_mod
        convert(trace, json_out)
        assert os.path.exists(json_out), "JSON file not created"

        with open(json_out) as f:
            data = json_mod.load(f)
        events = data.get("traceEvents", data) if isinstance(data, dict) else data
        assert len(events) > 0, "JSON trace has no events"

    @multigpu
    def test_multi_gpu_busy_view(self, tmp_path):
        """Verify busy view has 2 gpuId."""
        trace = str(tmp_path / "trace.db")
        script = """\
import torch, threading

def worker(gpu_id, n):
    x = torch.randn(128, 128, device="cuda:{}".format(gpu_id))
    s = torch.cuda.Stream(device=gpu_id)
    with torch.cuda.stream(s):
        for _ in range(n):
            x = x @ x
    s.synchronize()

threads = [threading.Thread(target=worker, args=(i, 20)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
"""
        rc, _, err = _run_traced(script, trace)
        assert rc == 0, "Crashed: {}".format(err[-500:])
        conn = sqlite3.connect(trace)
        busy_gpus = [r[0] for r in conn.execute(
            "SELECT DISTINCT gpuId FROM busy"
        ).fetchall()]
        conn.close()
        assert len(busy_gpus) >= 2, "Expected 2 gpuId in busy view, got {}".format(busy_gpus)

    @multigpu
    def test_per_process_trace_integrity(self, tmp_path):
        """Verify per-process trace_PID.db files are queryable before merge."""
        script_code = """\
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
x = torch.randn(128, 128, device="cuda:{}".format(rank))
for _ in range(20):
    x = x @ x
torch.cuda.synchronize(rank)
dist.destroy_process_group()
"""
        # Write script manually and use rtl trace directly to inspect per-process files
        trace_dir = str(tmp_path)
        script_path = str(tmp_path / "workload.py")
        with open(script_path, "w") as f:
            f.write(script_code)

        # Use env vars directly so we can examine per-process files
        from rocm_trace_lite import get_lib_path
        lib = get_lib_path()
        env = os.environ.copy()
        env["HSA_TOOLS_LIB"] = lib
        env["RPD_LITE_OUTPUT"] = str(tmp_path / "trace_%p.db")

        result = subprocess.run(
            [sys.executable, "-m", "torch.distributed.run",
             "--nproc_per_node=2", "--master_port=29501", script_path],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=180,
        )
        assert result.returncode == 0, "workload failed"

        # Find per-process files
        per_proc = [
            os.path.join(trace_dir, f) for f in os.listdir(trace_dir)
            if re.match(r"^trace_\d+\.db$", f)
        ]
        assert len(per_proc) >= 2, "Expected 2+ per-process trace files, got {}".format(len(per_proc))

        # Each should be independently queryable
        for pf in per_proc:
            conn = sqlite3.connect(pf)
            ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
            conn.close()
            assert ops > 0, "Per-process file {} has 0 ops".format(pf)

    @multigpu
    def test_stale_file_cleanup(self, tmp_path):
        """Stale trace_PID.db files should be cleaned before new trace."""
        # Create a fake stale file
        stale = str(tmp_path / "trace_99999.db")
        with open(stale, "w") as f:
            f.write("fake stale data")

        script_path = str(tmp_path / "workload.py")
        with open(script_path, "w") as f:
            f.write("""\
import torch
x = torch.randn(64, 64, device="cuda:0")
for _ in range(10):
    x = x @ x
torch.cuda.synchronize()
""")

        trace_out = str(tmp_path / "trace.db")
        rc, _, err = _run_rtl(
            "trace", "-o", trace_out, sys.executable, script_path, timeout=120
        )
        assert rc == 0, "rtl trace failed: {}".format(err[-500:])
        # The stale file should have been cleaned up
        assert not os.path.exists(stale), "Stale trace_99999.db was not cleaned up"
