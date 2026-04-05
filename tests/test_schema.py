"""T2 — TraceDB schema and data integrity tests.

Tests RPD database schema, record writing, and query correctness
using synthetic data. No GPU needed.
"""
import sqlite3
from conftest import SCHEMA_SQL, REQUIRED_TABLES, REQUIRED_VIEWS, populate_synthetic_trace


class TestSchema:
    """T2.1–T2.2 — Schema creation and compatibility."""

    def test_schema_creates_all_tables(self, tmp_rpd):
        """T2.1 — All required tables are created."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        for t in REQUIRED_TABLES:
            assert t in tables, f"Missing table: {t}"
        conn.close()

    def test_schema_creates_views(self, tmp_rpd):
        """T2.2a — top and busy views are created."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        views = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        ).fetchall()]
        for v in REQUIRED_VIEWS:
            assert v in views, f"Missing view: {v}"
        conn.close()

    def test_schema_survives_reopen(self, tmp_rpd):
        """T2.1b — Schema persists after close+reopen."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        conn.close()

        conn2 = sqlite3.connect(tmp_rpd)
        tables = [r[0] for r in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        for t in REQUIRED_TABLES:
            assert t in tables
        conn2.close()

    def test_schema_idempotent(self, tmp_rpd):
        """Creating schema twice should not fail or duplicate."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        conn.executescript(SCHEMA_SQL)  # second time
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert len(tables) == len(REQUIRED_TABLES)
        conn.close()


class TestKernelRecords:
    """T2.3 — Kernel op recording."""

    def test_kernel_record_count(self, tmp_rpd):
        """T2.3a — Correct number of ops written."""
        populate_synthetic_trace(tmp_rpd, num_kernels=50)
        conn = sqlite3.connect(tmp_rpd)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        assert count == 50
        conn.close()

    def test_kernel_timing_positive(self, tmp_rpd):
        """T2.3b — All ops have end > start."""
        populate_synthetic_trace(tmp_rpd, num_kernels=100)
        conn = sqlite3.connect(tmp_rpd)
        bad = conn.execute(
            "SELECT count(*) FROM rocpd_op WHERE end <= start"
        ).fetchone()[0]
        assert bad == 0, f"{bad} ops with end <= start"
        conn.close()

    def test_kernel_names_in_string_table(self, tmp_rpd):
        """T2.3c — Kernel names exist in rocpd_string."""
        populate_synthetic_trace(tmp_rpd, num_kernels=10)
        conn = sqlite3.connect(tmp_rpd)
        names = [r[0] for r in conn.execute(
            "SELECT DISTINCT s.string FROM rocpd_op o "
            "JOIN rocpd_string s ON o.description_id = s.id"
        ).fetchall()]
        assert len(names) > 0
        assert any(".kd" in n for n in names), f"No .kd kernel names: {names}"
        conn.close()

    def test_kernel_optype_set(self, tmp_rpd):
        """T2.3d — opType_id references 'KernelExecution'."""
        populate_synthetic_trace(tmp_rpd, num_kernels=10)
        conn = sqlite3.connect(tmp_rpd)
        types = [r[0] for r in conn.execute(
            "SELECT DISTINCT s.string FROM rocpd_op o "
            "JOIN rocpd_string s ON o.opType_id = s.id"
        ).fetchall()]
        assert "KernelExecution" in types
        conn.close()

    def test_gpu_id_set_correctly(self, tmp_rpd):
        """T2.3e — Multi-GPU ops have correct gpuId."""
        populate_synthetic_trace(tmp_rpd, num_kernels=100, num_gpus=4)
        conn = sqlite3.connect(tmp_rpd)
        gpu_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT gpuId FROM rocpd_op ORDER BY gpuId"
        ).fetchall()]
        assert gpu_ids == [0, 1, 2, 3]
        conn.close()


class TestStringDedup:
    """T2.7 — String table deduplication."""

    def test_string_dedup(self, tmp_rpd):
        """Same kernel name inserted many times -> 1 row in rocpd_string."""
        populate_synthetic_trace(tmp_rpd, num_kernels=500)
        conn = sqlite3.connect(tmp_rpd)
        # We have 5 unique kernel names + "KernelExecution"
        str_count = conn.execute("SELECT count(*) FROM rocpd_string").fetchone()[0]
        assert str_count == 6, f"Expected 6 strings, got {str_count}"
        conn.close()


class TestTopView:
    """T2 — top view correctness."""

    def test_top_view_returns_results(self, tmp_rpd):
        """top view should return aggregated kernel stats."""
        populate_synthetic_trace(tmp_rpd, num_kernels=100)
        conn = sqlite3.connect(tmp_rpd)
        rows = conn.execute("SELECT * FROM top").fetchall()
        assert len(rows) == 5, f"Expected 5 kernel types, got {len(rows)}"
        conn.close()

    def test_top_view_calls_correct(self, tmp_rpd):
        """top view Calls column should sum to total ops."""
        populate_synthetic_trace(tmp_rpd, num_kernels=100)
        conn = sqlite3.connect(tmp_rpd)
        total_calls = conn.execute("SELECT SUM(Calls) FROM top").fetchone()[0]
        assert total_calls == 100
        conn.close()

    def test_top_view_percentage_sums_to_100(self, tmp_rpd):
        """top view Pct column should sum to ~100%."""
        populate_synthetic_trace(tmp_rpd, num_kernels=100)
        conn = sqlite3.connect(tmp_rpd)
        total_pct = conn.execute("SELECT SUM(Pct) FROM top").fetchone()[0]
        assert abs(total_pct - 100.0) < 0.1, f"Pct sums to {total_pct}"
        conn.close()

    def test_top_view_ordered_by_total(self, tmp_rpd):
        """top view should be ordered by TotalNs descending."""
        populate_synthetic_trace(tmp_rpd, num_kernels=100)
        conn = sqlite3.connect(tmp_rpd)
        totals = [r[0] for r in conn.execute("SELECT TotalNs FROM top").fetchall()]
        assert totals == sorted(totals, reverse=True)
        conn.close()


class TestBusyView:
    """T2 — busy view correctness."""

    def test_busy_view_returns_results(self, tmp_rpd):
        """busy view should return per-GPU utilization."""
        populate_synthetic_trace(tmp_rpd, num_kernels=100, num_gpus=2)
        conn = sqlite3.connect(tmp_rpd)
        rows = conn.execute("SELECT * FROM busy").fetchall()
        assert len(rows) == 2, f"Expected 2 GPUs, got {len(rows)}"
        conn.close()

    def test_busy_view_utilization_bounded(self, tmp_rpd):
        """Utilization should be between 0 and 100%."""
        populate_synthetic_trace(tmp_rpd, num_kernels=100)
        conn = sqlite3.connect(tmp_rpd)
        util = conn.execute(
            "SELECT utilization_pct FROM busy"
        ).fetchone()[0]
        assert 0 < util <= 100, f"Utilization out of range: {util}"
        conn.close()


class TestEmptyTrace:
    """Edge case — empty trace handling."""

    def test_empty_trace_top_view(self, tmp_rpd):
        """top view on empty trace should return 0 rows, not error."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        rows = conn.execute("SELECT * FROM top").fetchall()
        assert len(rows) == 0
        conn.close()

    def test_empty_trace_busy_view(self, tmp_rpd):
        """busy view on empty trace should return 0 rows."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        rows = conn.execute("SELECT * FROM busy").fetchall()
        assert len(rows) == 0
        conn.close()
