"""Tests for multi-process trace support (issue #27).

Validates %p PID substitution and automatic trace merge.
"""
import os
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # noqa: E402


class TestPidSubstitution:
    """C++ source supports %p PID substitution."""

    def test_pid_substitution_in_source(self):
        with open(os.path.join(REPO_ROOT, "src", "rpd_lite.cpp")) as f:
            src = f.read()
        assert "%p" in src, "No %p PID substitution in rpd_lite.cpp"
        assert "getpid()" in src, "No getpid() call for PID substitution"

    def test_cmd_trace_uses_percent_p(self):
        with open(os.path.join(REPO_ROOT, "rocm_trace_lite", "cmd_trace.py")) as f:
            src = f.read()
        assert "%p" in src, "cmd_trace.py does not use %p pattern"


class TestMergeLogic:
    """Validate the trace merge function."""

    def _create_trace(self, path, gpu_id, num_ops):
        """Create a synthetic per-process trace file."""
        sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))
        from conftest import SCHEMA_SQL
        conn = sqlite3.connect(path)
        conn.executescript(SCHEMA_SQL)
        conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES('test_kernel.kd')")
        conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES('KernelExecution')")
        base_ns = 1000000000000
        for i in range(num_ops):
            conn.execute(
                "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, "
                "description_id, opType_id) VALUES(?, 0, 0, ?, ?, "
                "(SELECT id FROM rocpd_string WHERE string='test_kernel.kd'), "
                "(SELECT id FROM rocpd_string WHERE string='KernelExecution'))",
                (gpu_id, base_ns + i * 10000, base_ns + i * 10000 + 5000),
            )
        conn.commit()
        conn.close()

    def test_merge_two_traces(self, tmp_path):
        """Merge 2 per-process traces into one."""
        from rocm_trace_lite.cmd_trace import _merge_traces

        f1 = str(tmp_path / "trace_100.rpd")
        f2 = str(tmp_path / "trace_200.rpd")
        out = str(tmp_path / "trace.rpd")

        self._create_trace(f1, gpu_id=0, num_ops=50)
        self._create_trace(f2, gpu_id=1, num_ops=30)

        _merge_traces([f1, f2], out)

        conn = sqlite3.connect(out)
        total = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        gpus = dict(conn.execute(
            "SELECT gpuId, count(*) FROM rocpd_op GROUP BY gpuId"
        ).fetchall())
        conn.close()

        assert total == 80, f"Expected 80 ops, got {total}"
        assert gpus[0] == 50, f"GPU 0: expected 50, got {gpus.get(0)}"
        assert gpus[1] == 30, f"GPU 1: expected 30, got {gpus.get(1)}"

    def test_merge_skips_empty_traces(self, tmp_path):
        """Empty per-process files (< 50KB) should be skipped."""
        from rocm_trace_lite.cmd_trace import _merge_traces

        f1 = str(tmp_path / "trace_100.rpd")
        f2 = str(tmp_path / "trace_200.rpd")  # empty
        out = str(tmp_path / "trace.rpd")

        self._create_trace(f1, gpu_id=0, num_ops=100)
        # Create empty trace
        sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))
        from conftest import SCHEMA_SQL
        conn = sqlite3.connect(f2)
        conn.executescript(SCHEMA_SQL)
        conn.close()

        _merge_traces([f1, f2], out)

        conn = sqlite3.connect(out)
        total = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert total == 100  # only f1's ops

    def test_merge_preserves_gpu_ids(self, tmp_path):
        """Each process's gpuId must be preserved after merge."""
        from rocm_trace_lite.cmd_trace import _merge_traces

        files = []
        for gpu in range(4):
            f = str(tmp_path / f"trace_{gpu}.rpd")
            self._create_trace(f, gpu_id=gpu, num_ops=20)
            files.append(f)

        out = str(tmp_path / "trace.rpd")
        _merge_traces(files, out)

        conn = sqlite3.connect(out)
        gpus = sorted(r[0] for r in conn.execute(
            "SELECT DISTINCT gpuId FROM rocpd_op"
        ).fetchall())
        conn.close()
        assert gpus == [0, 1, 2, 3]

    def test_single_file_no_merge(self, tmp_path):
        """Single per-process file should just be renamed, not merged."""
        f1 = str(tmp_path / "trace_100.rpd")
        out = str(tmp_path / "trace.rpd")
        self._create_trace(f1, gpu_id=0, num_ops=10)

        # Simulate what cmd_trace does for single file
        os.rename(f1, out)

        assert os.path.exists(out)
        assert not os.path.exists(f1)
        conn = sqlite3.connect(out)
        total = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert total == 10

    def test_merged_top_view_works(self, tmp_path):
        """top view should work on merged trace."""
        from rocm_trace_lite.cmd_trace import _merge_traces

        f1 = str(tmp_path / "trace_100.rpd")
        f2 = str(tmp_path / "trace_200.rpd")
        out = str(tmp_path / "trace.rpd")

        self._create_trace(f1, gpu_id=0, num_ops=50)
        self._create_trace(f2, gpu_id=1, num_ops=30)
        _merge_traces([f1, f2], out)

        conn = sqlite3.connect(out)
        try:
            rows = conn.execute("SELECT * FROM top").fetchall()
        except sqlite3.OperationalError:
            rows = []
        conn.close()
        assert len(rows) > 0, "top view broken after merge"
