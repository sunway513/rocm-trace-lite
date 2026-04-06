"""Pre-release validation suite: microbench + E2E + artifact checks.

Run with: python3 -m pytest tests/test_release_validation.py -v --timeout=300
"""
import gzip
import json
import os
import re
import sqlite3
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "rocm_trace_lite", "lib", "librtl.so")

from conftest import _rocm_gpu_available, _rocm_gpu_count

HAS_GPU = _rocm_gpu_available()
GPU_COUNT = _rocm_gpu_count() if HAS_GPU else 0
gpu = pytest.mark.skipif(not HAS_GPU, reason="No GPU")
multigpu = pytest.mark.skipif(GPU_COUNT < 2, reason="Need 2+ GPUs")


def _run_rtl(tmp_path, script, name="trace", timeout=120, nproc=None):
    """Run script under rtl trace, return (trace_path, result)."""
    trace = str(tmp_path / "{}.db".format(name))
    script_path = str(tmp_path / "{}.py".format(name))
    with open(script_path, "w") as f:
        f.write(textwrap.dedent(script))
    cmd = [sys.executable, "-m", "rocm_trace_lite.cli", "trace", "-o", trace]
    if nproc:
        cmd += [sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node={}".format(nproc), "--master_port=29600",
                script_path]
    else:
        cmd += [sys.executable, script_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=timeout, cwd=REPO_ROOT)
    return trace, result


def _query(db, sql):
    conn = sqlite3.connect(db)
    val = conn.execute(sql).fetchone()
    conn.close()
    return val


def _get_elapsed(output):
    for line in output.splitlines():
        if "ELAPSED:" in line:
            return float(line.split(":")[1])
    return None


# =========================================================================
# SECTION 1: Microbenchmarks
# =========================================================================


class TestMicrobenchOverhead:
    """Measure profiler overhead on real GPU workloads."""

    @gpu
    def test_overhead_gemm_1k(self, tmp_path):
        """Profiler overhead on 1000 GEMM ops should be < 20%."""
        script = """\
        import torch, time
        x = torch.randn(256, 256, device="cuda:0")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            x = x @ x
        torch.cuda.synchronize()
        print("ELAPSED:{:.6f}".format(time.perf_counter() - t0))
        """
        script_path = str(tmp_path / "bench.py")
        with open(script_path, "w") as f:
            f.write(textwrap.dedent(script))
        r_base = subprocess.run([sys.executable, script_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=60, cwd=REPO_ROOT)
        base_time = _get_elapsed(r_base.stdout)
        assert base_time, "No baseline timing"

        trace, r_prof = _run_rtl(tmp_path, script, name="bench_prof")
        prof_time = _get_elapsed(r_prof.stdout)
        assert prof_time, "No profiled timing"

        overhead = (prof_time - base_time) / base_time * 100
        print("Base: {:.3f}s  Profiled: {:.3f}s  Overhead: {:.1f}%".format(
            base_time, prof_time, overhead))
        assert overhead < 20, "Overhead {:.1f}% exceeds 20%".format(overhead)

    @gpu
    def test_overhead_short_kernels(self, tmp_path):
        """Overhead on 5000 tiny kernels (worst case)."""
        script = """\
        import torch, time
        x = torch.randn(16, 16, device="cuda:0")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5000):
            x = x + 1
        torch.cuda.synchronize()
        print("ELAPSED:{:.6f}".format(time.perf_counter() - t0))
        """
        script_path = str(tmp_path / "short.py")
        with open(script_path, "w") as f:
            f.write(textwrap.dedent(script))
        r_base = subprocess.run([sys.executable, script_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=60, cwd=REPO_ROOT)
        base_time = _get_elapsed(r_base.stdout)
        trace, r_prof = _run_rtl(tmp_path, script, name="short_prof")
        prof_time = _get_elapsed(r_prof.stdout)
        overhead = (prof_time - base_time) / base_time * 100
        print("Base: {:.3f}s  Profiled: {:.3f}s  Overhead: {:.1f}%".format(
            base_time, prof_time, overhead))
        assert overhead < 50, "Overhead {:.1f}% exceeds 50%".format(overhead)

    @gpu
    def test_trace_file_size(self, tmp_path):
        """1000 ops should produce < 5MB trace."""
        trace, r = _run_rtl(tmp_path, """\
        import torch
        x = torch.randn(64, 64, device="cuda:0")
        for _ in range(1000):
            x = x @ x
        torch.cuda.synchronize()
        """)
        assert r.returncode == 0
        size_mb = os.path.getsize(trace) / 1024 / 1024
        print("Trace: {:.2f} MB for 1000 ops".format(size_mb))
        assert size_mb < 5


# =========================================================================
# SECTION 2: E2E single GPU
# =========================================================================


class TestE2ESingleGPU:
    """Real workload E2E on single GPU."""

    @gpu
    def test_gemm_sweep(self, tmp_path):
        """GEMM across 5 sizes, all captured."""
        trace, r = _run_rtl(tmp_path, """\
        import torch
        for n in [128, 256, 512, 1024, 2048]:
            a = torch.randn(n, n, device="cuda:0", dtype=torch.float16)
            c = a @ a
            torch.cuda.synchronize()
        """)
        assert r.returncode == 0
        ops = _query(trace, "SELECT count(*) FROM rocpd_op")[0]
        assert ops >= 5, "Got {} ops".format(ops)

    @gpu
    def test_conv2d(self, tmp_path):
        """Conv2D forward pass captured (tolerates MIOpen errors)."""
        trace, r = _run_rtl(tmp_path, """\
        import torch, torch.nn as nn
        model = nn.Conv2d(3, 16, 3, padding=1).cuda()
        x = torch.randn(1, 3, 16, 16, device="cuda:0")
        for _ in range(10):
            try:
                y = model(x)
            except RuntimeError:
                pass
        torch.cuda.synchronize()
        """)
        # Workload may fail (MIOpen issues) but profiler should still capture ops
        assert os.path.exists(trace), "No trace file"
        ops = _query(trace, "SELECT count(*) FROM rocpd_op")[0]
        assert ops >= 1, "Expected some ops, got {}".format(ops)

    @gpu
    def test_attention(self, tmp_path):
        """Multi-head attention captured."""
        trace, r = _run_rtl(tmp_path, """\
        import torch, torch.nn as nn
        attn = nn.MultiheadAttention(512, 8).cuda().half()
        q = k = v = torch.randn(32, 4, 512, device="cuda:0", dtype=torch.float16)
        for _ in range(5):
            out, _ = attn(q, k, v)
        torch.cuda.synchronize()
        """)
        assert r.returncode == 0
        assert _query(trace, "SELECT count(*) FROM rocpd_op")[0] >= 5

    @gpu
    def test_summary_in_stdout(self, tmp_path):
        """rtl trace stdout has kernel summary."""
        trace, r = _run_rtl(tmp_path, """\
        import torch
        x = torch.randn(512, 512, device="cuda:0")
        for _ in range(50):
            x = x @ x
        torch.cuda.synchronize()
        """)
        assert r.returncode == 0
        assert "GPU ops" in r.stdout

    @gpu
    def test_perfetto_json_valid(self, tmp_path):
        """Produces valid .json.gz Perfetto file."""
        trace, r = _run_rtl(tmp_path, """\
        import torch
        x = torch.randn(256, 256, device="cuda:0")
        for _ in range(20):
            x = x @ x
        torch.cuda.synchronize()
        """)
        assert r.returncode == 0
        json_gz = trace.replace(".db", ".json.gz")
        assert os.path.exists(json_gz)
        with gzip.open(json_gz, "rt") as f:
            data = json.load(f)
        assert isinstance(data, (list, dict))

    @gpu
    def test_roctx_markers(self, tmp_path):
        """roctx markers appear in trace."""
        trace, r = _run_rtl(tmp_path, """\
        import ctypes, os, torch
        lib = ctypes.CDLL(os.environ.get("HSA_TOOLS_LIB", "librtl.so"))
        lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        lib.roctxRangePop.restype = ctypes.c_int
        lib.roctxRangePushA(b"test_region")
        x = torch.randn(256, 256, device="cuda:0")
        y = x @ x
        torch.cuda.synchronize()
        lib.roctxRangePop()
        """)
        assert r.returncode == 0
        conn = sqlite3.connect(trace)
        markers = conn.execute(
            "SELECT count(*) FROM rocpd_op o JOIN rocpd_string s "
            "ON o.description_id=s.id WHERE s.string='test_region'"
        ).fetchone()[0]
        conn.close()
        assert markers >= 1

    @gpu
    def test_kernel_names_demangled(self, tmp_path):
        """Kernel names should not be hex addresses."""
        trace, r = _run_rtl(tmp_path, """\
        import torch
        x = torch.randn(512, 512, device="cuda:0")
        y = x @ x
        torch.cuda.synchronize()
        """)
        assert r.returncode == 0
        conn = sqlite3.connect(trace)
        names = [r[0] for r in conn.execute(
            "SELECT s.string FROM rocpd_op o JOIN rocpd_string s "
            "ON o.description_id=s.id WHERE o.gpuId >= 0"
        ).fetchall()]
        conn.close()
        hex_only = [n for n in names if n.startswith("kernel_0x")]
        assert len(hex_only) < len(names), "All kernel names are hex fallbacks"


# =========================================================================
# SECTION 3: E2E multi GPU
# =========================================================================


class TestE2EMultiGPU:
    """Multi-GPU E2E tests."""

    @multigpu
    def test_2gpu_merge(self, tmp_path):
        """2 GPU torchrun, merged trace has both gpuIds."""
        trace, r = _run_rtl(tmp_path, """\
        import torch, torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        x = torch.randn(256, 256, device="cuda:{}".format(rank))
        for _ in range(10):
            x = x @ x
        torch.cuda.synchronize()
        dist.destroy_process_group()
        """, nproc=2, timeout=180)
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        conn = sqlite3.connect(trace)
        gpu_ids = set(r[0] for r in conn.execute(
            "SELECT DISTINCT gpuId FROM rocpd_op WHERE gpuId >= 0").fetchall())
        conn.close()
        assert len(gpu_ids) >= 2

    @multigpu
    def test_8gpu_full(self, tmp_path):
        """8 GPU torchrun."""
        if GPU_COUNT < 8:
            pytest.skip("Need 8 GPUs")
        trace, r = _run_rtl(tmp_path, """\
        import torch, torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        x = torch.randn(64, 64, device="cuda:{}".format(rank))
        for _ in range(5):
            x = x @ x
        torch.cuda.synchronize()
        dist.destroy_process_group()
        """, nproc=8, timeout=180)
        assert r.returncode == 0, "Failed: {}".format(r.stderr[-500:])
        conn = sqlite3.connect(trace)
        gpu_ids = set(r[0] for r in conn.execute(
            "SELECT DISTINCT gpuId FROM rocpd_op WHERE gpuId >= 0").fetchall())
        conn.close()
        assert len(gpu_ids) >= 8

    @multigpu
    def test_nccl_allreduce(self, tmp_path):
        """NCCL all-reduce kernels captured."""
        trace, r = _run_rtl(tmp_path, """\
        import torch, torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        x = torch.randn(1024, device="cuda:{}".format(rank))
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.destroy_process_group()
        """, nproc=2, timeout=180)
        assert r.returncode == 0
        ops = _query(trace, "SELECT count(*) FROM rocpd_op WHERE gpuId >= 0")[0]
        assert ops > 0


# =========================================================================
# SECTION 4: Release artifact validation
# =========================================================================


class TestReleaseArtifacts:
    """Validate built artifacts."""

    def test_no_forbidden_deps(self):
        """librtl.so must not directly link roctracer/rocprofiler/libroctx64."""
        if not os.path.exists(LIB_PATH):
            pytest.skip("librtl.so not built")
        # Check NEEDED entries (direct deps), not transitive via ldd
        r = subprocess.run(["readelf", "-d", LIB_PATH], stdout=subprocess.PIPE,
                           universal_newlines=True)
        for dep in ["roctracer", "rocprofiler", "libroctx64"]:
            assert dep not in r.stdout, "Direct dep on {}".format(dep)

    def test_exports_onload(self):
        if not os.path.exists(LIB_PATH):
            pytest.skip("librtl.so not built")
        r = subprocess.run(["nm", "-D", LIB_PATH], stdout=subprocess.PIPE,
                           universal_newlines=True)
        assert "T OnLoad" in r.stdout
        assert "T OnUnload" in r.stdout

    def test_exports_all_roctx(self):
        if not os.path.exists(LIB_PATH):
            pytest.skip("librtl.so not built")
        r = subprocess.run(["nm", "-D", LIB_PATH], stdout=subprocess.PIPE,
                           universal_newlines=True)
        for sym in ["roctxRangePushA", "roctxRangePop", "roctxMarkA",
                    "roctxRangeStartA", "roctxRangeStop"]:
            assert sym in r.stdout, "Missing: {}".format(sym)

    def test_version_consistency(self):
        from rocm_trace_lite import __version__
        pyproject = os.path.join(REPO_ROOT, "pyproject.toml")
        with open(pyproject) as f:
            match = re.search(r'version = "([^"]+)"', f.read())
        assert match
        assert __version__ == match.group(1)

    def test_cli_version(self):
        from rocm_trace_lite import __version__
        r = subprocess.run([sys.executable, "-m", "rocm_trace_lite.cli", "--version"],
                           stdout=subprocess.PIPE, universal_newlines=True)
        assert __version__ in r.stdout
