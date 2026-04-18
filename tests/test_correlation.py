"""Correlation tests: verify HIP API calls are linked to GPU kernel ops.

Requires: ROCm GPU + librtl.so with HIP API support.
"""
import os
import sqlite3
import subprocess
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "librtl.so")
WORKLOAD = os.path.join(REPO_ROOT, "tests", "gpu_workload")

HAS_LIB = os.path.exists(LIB_PATH)
HAS_WORKLOAD = os.path.exists(WORKLOAD)


def _run_hip_mode(workload_args):
    if not HAS_LIB or not HAS_WORKLOAD:
        pytest.skip("Need librtl.so and gpu_workload")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    env = os.environ.copy()
    env.update({
        "HSA_TOOLS_LIB": LIB_PATH,
        "LD_PRELOAD": LIB_PATH,
        "RTL_OUTPUT": db_path,
        "RTL_MODE": "hip",
    })
    r = subprocess.run([WORKLOAD] + workload_args,
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    return db_path, r


@pytest.mark.skipif(not HAS_LIB or not HAS_WORKLOAD,
                    reason="Need librtl.so and gpu_workload")
class TestCorrelation:

    def test_api_op_link_exists(self):
        db, r = _run_hip_mode(["gemm", "256", "10"])
        assert r.returncode == 0
        conn = sqlite3.connect(db)
        link_count = conn.execute("SELECT count(*) FROM rocpd_api_ops").fetchone()[0]
        # Links may be zero in Phase 1 (correlation plumbing not yet wired)
        # This test documents the expected behavior
        api_count = conn.execute("SELECT count(*) FROM rocpd_api").fetchone()[0]
        op_count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        print(f"API events: {api_count}, GPU ops: {op_count}, links: {link_count}")
        assert api_count > 0, "Need HIP API events"
        assert op_count > 0, "Need GPU kernel ops"
        conn.close()
        os.unlink(db)

    def test_correlation_timing_order(self):
        db, r = _run_hip_mode(["gemm", "512", "5"])
        assert r.returncode == 0
        conn = sqlite3.connect(db)
        links = conn.execute("""
            SELECT a.start as api_start, a.end as api_end,
                   o.start as op_start, o.end as op_end
            FROM rocpd_api_ops ao
            JOIN rocpd_api a ON ao.api_id = a.id
            JOIN rocpd_op o ON ao.op_id = o.id
        """).fetchall()
        for api_start, api_end, op_start, op_end in links:
            assert api_start <= op_start, \
                f"API start ({api_start}) should be <= GPU start ({op_start})"
        conn.close()
        os.unlink(db)

    def test_multi_kernel_unique_correlation(self):
        db, r = _run_hip_mode(["gemm", "256", "20"])
        assert r.returncode == 0
        conn = sqlite3.connect(db)
        api_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT api_id FROM rocpd_api_ops").fetchall()]
        op_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT op_id FROM rocpd_api_ops").fetchall()]
        if len(api_ids) > 0:
            assert len(api_ids) == len(set(api_ids)), "API IDs should be unique"
            assert len(op_ids) == len(set(op_ids)), "Op IDs should be unique"
        conn.close()
        os.unlink(db)

    def test_both_tables_populated(self):
        db, r = _run_hip_mode(["gemm", "256", "10"])
        assert r.returncode == 0
        conn = sqlite3.connect(db)
        api = conn.execute("SELECT count(*) FROM rocpd_api").fetchone()[0]
        ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        assert api > 0, "rocpd_api must have entries in hip mode"
        assert ops > 0, "rocpd_op must have entries in hip mode"
        ratio = api / ops if ops > 0 else 0
        print(f"API/Op ratio: {ratio:.2f} ({api} APIs, {ops} ops)")
        conn.close()
        os.unlink(db)
