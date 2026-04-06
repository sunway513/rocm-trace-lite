"""CPU-only tests for cmd_trace.py: _generate_summary, _generate_perfetto, _merge_traces, _checkpoint_wal."""
import gzip
import json
import os
import sqlite3

from rocm_trace_lite.cmd_trace import (
    _checkpoint_wal,
    _generate_perfetto,
    _generate_summary,
    _merge_traces,
)

# Re-use shared schema
from conftest import SCHEMA_SQL, populate_synthetic_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_empty_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    conn.close()
    return path


def _make_single_kernel_db(path, kernel_name="my_kernel.kd", gpu_id=0,
                           start=1000000000, duration=5000, count=1):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)", (kernel_name,))
    conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)", ("KernelExecution",))
    for i in range(count):
        s = start + i * 10000
        e = s + duration
        conn.execute(
            "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, description_id, opType_id) "
            "VALUES(?, 0, 0, ?, ?, "
            "(SELECT id FROM rocpd_string WHERE string=?), "
            "(SELECT id FROM rocpd_string WHERE string='KernelExecution'))",
            (gpu_id, s, e, kernel_name),
        )
    conn.commit()
    conn.close()
    return path


# ===========================================================================
# _generate_summary
# ===========================================================================

class TestGenerateSummary:

    def test_empty_trace(self, tmp_path):
        db = str(tmp_path / "empty.db")
        _make_empty_db(db)
        result = _generate_summary(db)
        assert result is not None
        assert "0 GPU ops" in result

    def test_single_kernel(self, tmp_path):
        db = str(tmp_path / "single.db")
        _make_single_kernel_db(db, kernel_name="gemm_kernel.kd", count=3)
        result = _generate_summary(db)
        assert "gemm_kernel.kd" in result
        assert "3" in result  # calls count

    def test_multi_gpu(self, tmp_path):
        db = str(tmp_path / "multigpu.db")
        populate_synthetic_trace(db, num_kernels=20, num_gpus=2)
        result = _generate_summary(db)
        assert "GPU 0" in result
        assert "GPU 1" in result

    def test_large_trace_top20(self, tmp_path):
        db = str(tmp_path / "large.db")
        conn = sqlite3.connect(db)
        conn.executescript(SCHEMA_SQL)
        conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)", ("KernelExecution",))
        # Create 30 distinct kernels, each with ops
        for k in range(30):
            name = "kernel_{:03d}.kd".format(k)
            conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)", (name,))
            for i in range(5):
                s = 1000000000 + k * 100000 + i * 10000
                e = s + 5000
                conn.execute(
                    "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, description_id, opType_id) "
                    "VALUES(0, 0, 0, ?, ?, "
                    "(SELECT id FROM rocpd_string WHERE string=?), "
                    "(SELECT id FROM rocpd_string WHERE string='KernelExecution'))",
                    (s, e, name),
                )
        conn.commit()
        conn.close()

        result = _generate_summary(db)
        assert result is not None
        # top view is LIMIT 20, so at most 20 kernel lines in the table
        # Count data lines (lines with ".kd" in them)
        kernel_lines = [line for line in result.splitlines() if ".kd" in line]
        assert len(kernel_lines) <= 20


# ===========================================================================
# _generate_perfetto
# ===========================================================================

class TestGeneratePerfetto:

    def test_normal_trace(self, tmp_path):
        db = str(tmp_path / "trace.db")
        populate_synthetic_trace(db, num_kernels=10, num_gpus=1)
        out = str(tmp_path / "trace.json.gz")
        _generate_perfetto(db, out)
        assert os.path.exists(out)
        with gzip.open(out, "rb") as f:
            data = json.loads(f.read())
        assert "traceEvents" in data
        assert len(data["traceEvents"]) > 0

    def test_empty_trace(self, tmp_path):
        db = str(tmp_path / "empty.db")
        _make_empty_db(db)
        out = str(tmp_path / "empty.json.gz")
        _generate_perfetto(db, out)
        # Empty trace: convert() prints "trace is empty" and returns without writing.
        # _generate_perfetto catches the exception or the file may not exist.
        # Either way, should not crash.

    def test_output_path_nonexistent_dir(self, tmp_path):
        db = str(tmp_path / "trace.db")
        populate_synthetic_trace(db, num_kernels=5, num_gpus=1)
        out = str(tmp_path / "no_such_dir" / "trace.json.gz")
        # Should not crash — the function catches exceptions
        _generate_perfetto(db, out)


# ===========================================================================
# _merge_traces
# ===========================================================================

class TestMergeTraces:

    def test_single_file(self, tmp_path):
        db = str(tmp_path / "single.db")
        populate_synthetic_trace(db, num_kernels=10, num_gpus=1)
        out = str(tmp_path / "merged.db")
        _merge_traces([db], out)
        assert os.path.exists(out)
        conn = sqlite3.connect(out)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert count == 10

    def test_two_files(self, tmp_path):
        db1 = str(tmp_path / "trace1.db")
        db2 = str(tmp_path / "trace2.db")
        _make_single_kernel_db(db1, kernel_name="kernel_a.kd", count=5,
                               start=1000000000)
        _make_single_kernel_db(db2, kernel_name="kernel_b.kd", count=3,
                               start=2000000000)
        out = str(tmp_path / "merged.db")
        _merge_traces([db1, db2], out)
        conn = sqlite3.connect(out)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert count == 8

    def test_empty_source(self, tmp_path):
        db1 = str(tmp_path / "normal.db")
        db2 = str(tmp_path / "empty.db")
        _make_single_kernel_db(db1, kernel_name="real_kernel.kd", count=5)
        _make_empty_db(db2)
        out = str(tmp_path / "merged.db")
        _merge_traces([db1, db2], out)
        conn = sqlite3.connect(out)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert count == 5

    def test_corrupted_source(self, tmp_path):
        db1 = str(tmp_path / "normal.db")
        bad = str(tmp_path / "corrupted.db")
        _make_single_kernel_db(db1, kernel_name="good_kernel.kd", count=5)
        with open(bad, "w") as f:
            f.write("this is not a sqlite file")
        out = str(tmp_path / "merged.db")
        _merge_traces([db1, bad], out)
        conn = sqlite3.connect(out)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert count == 5

    def test_string_dedup(self, tmp_path):
        db1 = str(tmp_path / "a.db")
        db2 = str(tmp_path / "b.db")
        shared_name = "shared_kernel.kd"
        _make_single_kernel_db(db1, kernel_name=shared_name, count=2,
                               start=1000000000)
        _make_single_kernel_db(db2, kernel_name=shared_name, count=3,
                               start=2000000000)
        out = str(tmp_path / "merged.db")
        _merge_traces([db1, db2], out)
        conn = sqlite3.connect(out)
        dup_count = conn.execute(
            "SELECT count(*) FROM rocpd_string WHERE string=?", (shared_name,)
        ).fetchone()[0]
        ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert dup_count == 1
        assert ops == 5


# ===========================================================================
# _checkpoint_wal
# ===========================================================================

class TestCheckpointWal:

    def test_wal_mode_db(self, tmp_path):
        db = str(tmp_path / "wal.db")
        conn = sqlite3.connect(db)
        conn.executescript(SCHEMA_SQL)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("INSERT INTO rocpd_string(string) VALUES('test')")
        conn.commit()
        conn.close()

        _checkpoint_wal(db)

        conn = sqlite3.connect(db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "delete"

    def test_nonexistent_db(self, tmp_path):
        db = str(tmp_path / "does_not_exist.db")
        # Should not crash
        _checkpoint_wal(db)

    def test_corrupted_db(self, tmp_path):
        bad = str(tmp_path / "corrupted.db")
        with open(bad, "w") as f:
            f.write("not a database")
        # Should not crash
        _checkpoint_wal(bad)
