"""Combined E2E test: multi-thread + multi-stream + multi-GPU + HIP Graph + roctx.

This is the integration stress test that validates all features work together.
Requires a ROCm GPU (ideally >=2 GPUs).
"""
import os
import sqlite3
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "librtl.so")


def _has_gpu():
    from conftest import _rocm_gpu_available
    return _rocm_gpu_available()


def _skip_if_no_gpu():
    if not _has_gpu() or not os.path.exists(LIB_PATH):
        import pytest
        pytest.skip("No GPU or librtl.so not built")


def _gpu_count():
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        return int(r.stdout.decode().strip())
    except Exception:
        return 0


def _run_script(script_path, trace_path, timeout=180):
    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = LIB_PATH
    env["RPD_LITE_OUTPUT"] = trace_path
    r = subprocess.run(
        [sys.executable, script_path],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout
    )
    return r.returncode, r.stdout.decode(), r.stderr.decode()


COMBINED_SCRIPT = '''
import ctypes
import os
import threading
import torch

# Load roctx shim
lib_path = os.environ.get("HSA_TOOLS_LIB", "librtl.so")
try:
    roctx = ctypes.CDLL(lib_path)
    roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
    roctx.roctxRangePushA.restype = ctypes.c_int
    roctx.roctxRangePop.restype = ctypes.c_int
    roctx.roctxMarkA.argtypes = [ctypes.c_char_p]
    HAS_ROCTX = True
except Exception:
    HAS_ROCTX = False

num_gpus = min(torch.cuda.device_count(), {num_gpus})
errors = []

# Phase 1: Multi-GPU eager with roctx markers
def eager_worker(gpu_id, stream_id):
    try:
        if HAS_ROCTX:
            roctx.roctxRangePushA(f"eager_gpu{{gpu_id}}_s{{stream_id}}".encode())
        stream = torch.cuda.Stream(device=gpu_id)
        x = torch.randn(128, 128, device=f"cuda:{{gpu_id}}")
        with torch.cuda.stream(stream):
            for _ in range(20):
                x = x @ x
        stream.synchronize()
        if HAS_ROCTX:
            roctx.roctxRangePop()
    except Exception as e:
        errors.append(f"eager gpu{{gpu_id}} s{{stream_id}}: {{e}}")

if HAS_ROCTX:
    roctx.roctxRangePushA(b"phase1_eager")

threads = []
for gpu in range(num_gpus):
    for sid in range(2):
        t = threading.Thread(target=eager_worker, args=(gpu, sid))
        threads.append(t)
for t in threads:
    t.start()
for t in threads:
    t.join()

if HAS_ROCTX:
    roctx.roctxRangePop()
    roctx.roctxMarkA(b"phase1_done")

for gpu in range(num_gpus):
    torch.cuda.synchronize(gpu)

# Phase 2: HIP Graph capture + replay on GPU 0
if HAS_ROCTX:
    roctx.roctxRangePushA(b"phase2_graph")

x = torch.randn(256, 256, device="cuda:0")
s = torch.cuda.Stream(device=0)
with torch.cuda.stream(s):
    _ = x @ x
s.synchronize()

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s):
    y = x @ x
for _ in range(30):
    g.replay()
s.synchronize()

if HAS_ROCTX:
    roctx.roctxRangePop()
    roctx.roctxMarkA(b"phase2_done")

# Phase 3: Mixed — eager on multiple streams WHILE graph replays
if HAS_ROCTX:
    roctx.roctxRangePushA(b"phase3_mixed")

def mixed_eager(gpu_id):
    try:
        stream = torch.cuda.Stream(device=gpu_id)
        x = torch.randn(128, 128, device=f"cuda:{{gpu_id}}")
        with torch.cuda.stream(stream):
            for _ in range(10):
                x = x @ x
        stream.synchronize()
    except Exception as e:
        errors.append(f"mixed gpu{{gpu_id}}: {{e}}")

# Graph replay + eager concurrently
graph_thread = threading.Thread(target=lambda: [g.replay() for _ in range(20)])
eager_threads = [threading.Thread(target=mixed_eager, args=(i % num_gpus,)) for i in range(4)]

graph_thread.start()
for t in eager_threads:
    t.start()
graph_thread.join()
for t in eager_threads:
    t.join()

for gpu in range(num_gpus):
    torch.cuda.synchronize(gpu)

if HAS_ROCTX:
    roctx.roctxRangePop()
    roctx.roctxMarkA(b"all_done")

if errors:
    for e in errors:
        print(f"ERROR: {{e}}")
    raise RuntimeError(f"{{len(errors)}} errors")
print("ALL PHASES COMPLETE")
'''


class TestCombinedE2E:
    """All features combined: multi-thread + multi-stream + multi-GPU + graph + roctx."""

    def test_full_combined(self, tmp_path):
        """The big integration test."""
        _skip_if_no_gpu()
        num_gpus = min(_gpu_count(), 4)
        if num_gpus < 1:
            import pytest
            pytest.skip("No GPU")

        script = tmp_path / "combined.py"
        script.write_text(COMBINED_SCRIPT.format(num_gpus=num_gpus))
        trace = str(tmp_path / "combined.db")

        rc, stdout, stderr = _run_script(str(script), trace)
        assert rc == 0, f"Combined test crashed:\nstdout: {stdout[-500:]}\nstderr: {stderr[-500:]}"
        assert "ALL PHASES COMPLETE" in stdout

    def test_combined_has_kernel_ops(self, tmp_path):
        """Trace must contain kernel dispatch ops."""
        _skip_if_no_gpu()
        num_gpus = min(_gpu_count(), 4)
        script = tmp_path / "combined.py"
        script.write_text(COMBINED_SCRIPT.format(num_gpus=num_gpus))
        trace = str(tmp_path / "combined.db")

        rc, _, _ = _run_script(str(script), trace)
        assert rc == 0

        conn = sqlite3.connect(trace)
        ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # Phase1: num_gpus*2*20=80+ GEMMs, Phase2: 30+ replays, Phase3: 20+40
        assert ops >= 50, f"Expected >=50 ops, got {ops}"

    def test_combined_has_roctx_markers(self, tmp_path):
        """Trace must contain roctx UserMarker records."""
        _skip_if_no_gpu()
        num_gpus = min(_gpu_count(), 4)
        script = tmp_path / "combined.py"
        script.write_text(COMBINED_SCRIPT.format(num_gpus=num_gpus))
        trace = str(tmp_path / "combined.db")

        rc, _, _ = _run_script(str(script), trace)
        assert rc == 0

        conn = sqlite3.connect(trace)
        markers = [r[0] for r in conn.execute(
            "SELECT s.string FROM rocpd_op o "
            "JOIN rocpd_string s ON o.description_id = s.id "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'UserMarker'"
        ).fetchall()]
        conn.close()

        assert "phase1_eager" in markers, f"Missing phase1 marker. Got: {markers}"
        assert "phase2_graph" in markers, f"Missing phase2 marker. Got: {markers}"
        assert "phase3_mixed" in markers, f"Missing phase3 marker. Got: {markers}"
        assert "all_done" in markers, f"Missing all_done marker. Got: {markers}"

    def test_combined_multi_gpu_ops(self, tmp_path):
        """Multi-GPU workload should show ops on multiple gpuIds."""
        _skip_if_no_gpu()
        num_gpus = min(_gpu_count(), 4)
        if num_gpus < 2:
            import pytest
            pytest.skip("Need >=2 GPUs")

        script = tmp_path / "combined.py"
        script.write_text(COMBINED_SCRIPT.format(num_gpus=num_gpus))
        trace = str(tmp_path / "combined.db")

        rc, _, _ = _run_script(str(script), trace)
        assert rc == 0

        conn = sqlite3.connect(trace)
        gpu_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT gpuId FROM rocpd_op WHERE gpuId >= 0"
        ).fetchall()]
        conn.close()

        assert len(gpu_ids) >= 2, f"Expected ops on >=2 GPUs, got gpuIds: {gpu_ids}"

    def test_combined_no_data_loss(self, tmp_path):
        """No kernel dispatches should be silently dropped."""
        _skip_if_no_gpu()
        num_gpus = min(_gpu_count(), 2)
        script = tmp_path / "combined.py"
        script.write_text(COMBINED_SCRIPT.format(num_gpus=num_gpus))
        trace = str(tmp_path / "combined.db")

        rc, _, _ = _run_script(str(script), trace)
        assert rc == 0

        conn = sqlite3.connect(trace)
        ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        # Count kernels only (exclude UserMarker)
        kernels = conn.execute(
            "SELECT count(*) FROM rocpd_op o "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'KernelExecution'"
        ).fetchone()[0]
        markers = conn.execute(
            "SELECT count(*) FROM rocpd_op o "
            "JOIN rocpd_string ot ON o.opType_id = ot.id "
            "WHERE ot.string = 'UserMarker'"
        ).fetchone()[0]
        conn.close()

        assert kernels > 0, "No kernel ops captured"
        assert markers > 0, "No roctx markers captured"
        assert ops == kernels + markers, f"Ops mismatch: {ops} != {kernels} + {markers}"

    def test_combined_top_view_works(self, tmp_path):
        """top view should work on combined trace."""
        _skip_if_no_gpu()
        num_gpus = min(_gpu_count(), 2)
        script = tmp_path / "combined.py"
        script.write_text(COMBINED_SCRIPT.format(num_gpus=num_gpus))
        trace = str(tmp_path / "combined.db")

        rc, _, _ = _run_script(str(script), trace)
        assert rc == 0

        conn = sqlite3.connect(trace)
        top = conn.execute("SELECT * FROM top LIMIT 5").fetchall()
        conn.close()
        assert len(top) > 0, "top view returned no rows"
        # Top kernel should be a GEMM (Cijk) — UserMarkers are excluded from top view
        all_names = [str(r[0]) for r in top]
        assert any("Cijk" in n for n in all_names), f"No GEMM in top 5: {[n[:40] for n in all_names]}"

    def test_combined_convert_to_json(self, tmp_path):
        """Convert combined trace to Perfetto JSON."""
        _skip_if_no_gpu()
        num_gpus = min(_gpu_count(), 2)
        script = tmp_path / "combined.py"
        script.write_text(COMBINED_SCRIPT.format(num_gpus=num_gpus))
        trace = str(tmp_path / "combined.db")
        json_out = str(tmp_path / "combined.json")

        rc, _, _ = _run_script(str(script), trace)
        assert rc == 0

        # Convert
        import json as json_mod
        sys.path.insert(0, REPO_ROOT)
        from rocm_trace_lite.cmd_convert import convert
        convert(trace, json_out)

        assert os.path.exists(json_out)
        with open(json_out) as f:
            data = json_mod.load(f)
        events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
        assert len(events) > 0, "No events in JSON"
