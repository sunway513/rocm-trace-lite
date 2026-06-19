"""T-roctx — rocpd_op.roctxId column + roctx-range attribution.

The op->roctx feature tags each GPU op with its enclosing roctx range's id (`roctxId`), and marker
rows (gpuId<0) store their own id, so a kernel attributes to its range via
`kernel.roctxId == marker.roctxId` -- no timestamp bucketing. These tests use synthetic data that
mirrors exactly what the patched tracer writes, so they run on CPU (no GPU) in CI.
"""
import sqlite3

from conftest import SCHEMA_SQL


def _populate_roctx_trace(path):
    """Mirror the patched tracer's output: two marker rows whose roctxId is their own id, and
    kernels tagged with the enclosing range's id. layer_0 (id=100): 4x incr; layer_1 (id=200): 2x dbl."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    for s in ("incr", "dbl", "layer_0", "layer_1", "KernelExecution", "UserMarker"):
        conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)", (s,))

    def op(gpu, name, optype, start, end, roctx):
        conn.execute(
            "INSERT INTO rocpd_op(gpuId,queueId,sequenceId,start,end,description_id,opType_id,roctxId) "
            "VALUES(?,?,0,?,?,(SELECT id FROM rocpd_string WHERE string=?),"
            "(SELECT id FROM rocpd_string WHERE string=?),?)",
            (gpu, 0, start, end, name, optype, roctx))

    op(-1, "layer_0", "UserMarker", 1000, 2000, 100)   # marker stores its own id
    op(-1, "layer_1", "UserMarker", 2000, 3000, 200)
    t = 1000
    for _ in range(4):
        op(0, "incr", "KernelExecution", t, t + 50, 100); t += 100   # tagged with layer_0's id
    for _ in range(2):
        op(0, "dbl", "KernelExecution", t, t + 50, 200); t += 100    # tagged with layer_1's id
    conn.commit()
    conn.close()


class TestRoctxIdColumn:
    def test_rocpd_op_has_roctxid_column(self, tmp_rpd):
        """The schema exposes the new roctxId column on rocpd_op."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(rocpd_op)")]
        conn.close()
        assert "roctxId" in cols, f"roctxId missing from rocpd_op: {cols}"

    def test_default_roctxid_is_zero(self, tmp_rpd):
        """An op inserted without roctxId defaults to 0 (back-compat: old inserts omit it)."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES('k')")
        conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES('KernelExecution')")
        conn.execute(
            "INSERT INTO rocpd_op(gpuId,queueId,sequenceId,start,end,description_id,opType_id) "
            "VALUES(0,0,0,10,20,(SELECT id FROM rocpd_string WHERE string='k'),"
            "(SELECT id FROM rocpd_string WHERE string='KernelExecution'))")
        conn.commit()
        val = conn.execute("SELECT roctxId FROM rocpd_op").fetchone()[0]
        conn.close()
        assert val == 0, f"expected default 0, got {val}"


class TestRoctxAttribution:
    def test_every_kernel_links_to_its_range(self, tmp_rpd):
        """Each kernel joins to exactly its enclosing marker via roctxId, with the right region."""
        _populate_roctx_trace(tmp_rpd)
        conn = sqlite3.connect(tmp_rpd)
        rows = conn.execute(
            "SELECT ks.string, ms.string "
            "FROM rocpd_op k JOIN rocpd_string ks ON k.description_id=ks.id "
            "JOIN rocpd_op m ON m.roctxId=k.roctxId AND m.gpuId<0 "
            "JOIN rocpd_string ms ON m.description_id=ms.id "
            "WHERE k.gpuId>=0").fetchall()
        conn.close()
        assert len(rows) == 6, f"expected 6 linked kernels, got {len(rows)}"
        expect = {"incr": "layer_0", "dbl": "layer_1"}
        for kernel, region in rows:
            assert region == expect[kernel], f"{kernel} -> {region}, expected {expect[kernel]}"

    def test_attribution_counts(self, tmp_rpd):
        """Per-range kernel counts match the ground truth."""
        _populate_roctx_trace(tmp_rpd)
        conn = sqlite3.connect(tmp_rpd)
        counts = dict(conn.execute(
            "SELECT ms.string, COUNT(*) "
            "FROM rocpd_op k JOIN rocpd_op m ON m.roctxId=k.roctxId AND m.gpuId<0 "
            "JOIN rocpd_string ms ON m.description_id=ms.id "
            "WHERE k.gpuId>=0 GROUP BY ms.string").fetchall())
        conn.close()
        assert counts == {"layer_0": 4, "layer_1": 2}, counts
