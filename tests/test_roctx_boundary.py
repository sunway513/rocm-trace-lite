"""Roctx boundary conditions, symbol export guard, converter consistency, kernel name fallback."""
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from conftest import SCHEMA_SQL, populate_synthetic_trace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RPD2TRACE = os.path.join(REPO_ROOT, "tools", "rpd2trace.py")


def _find_librtl():
    """Return path to librtl.so or None if not found."""
    candidates = [
        os.path.join(REPO_ROOT, "librtl.so"),
        os.path.join(REPO_ROOT, "build", "librtl.so"),
        os.path.join(REPO_ROOT, "rocm_trace_lite", "lib", "librtl.so"),
        "/usr/local/lib/librtl.so",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


LIBRTL_PATH = _find_librtl()
SKIP_NO_SO = pytest.mark.skipif(
    LIBRTL_PATH is None,
    reason="librtl.so not found (not built)",
)


def insert_roctx_record(conn, message, start_ns, duration_ns, correlation_id=1):
    """Simulate what roctx_shim.cpp writes to the DB."""
    conn.execute(
        "INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)", (message,)
    )
    conn.execute(
        "INSERT OR IGNORE INTO rocpd_string(string) VALUES('UserMarker')"
    )
    conn.execute(
        "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, "
        "description_id, opType_id) VALUES(-1, 0, 0, ?, ?, "
        "(SELECT id FROM rocpd_string WHERE string=?), "
        "(SELECT id FROM rocpd_string WHERE string='UserMarker'))",
        (start_ns, start_ns + duration_ns, message),
    )


def insert_kernel_record(conn, name, gpu_id, start_ns, duration_ns):
    """Insert a synthetic kernel op."""
    conn.execute(
        "INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)", (name,)
    )
    conn.execute(
        "INSERT OR IGNORE INTO rocpd_string(string) VALUES('KernelExecution')"
    )
    conn.execute(
        "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, "
        "description_id, opType_id) VALUES(?, 0, 0, ?, ?, "
        "(SELECT id FROM rocpd_string WHERE string=?), "
        "(SELECT id FROM rocpd_string WHERE string='KernelExecution'))",
        (gpu_id, start_ns, start_ns + duration_ns, name),
    )


# ---------------------------------------------------------------------------
# Roctx Boundary Tests
# ---------------------------------------------------------------------------

class TestRoctxBoundary:

    def test_empty_message_roctx(self, tmp_rpd):
        """Empty string message should be stored and queryable without crash."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        insert_roctx_record(conn, "", 1000000, 5000)
        conn.commit()

        row = conn.execute(
            "SELECT s.string FROM rocpd_op o "
            "JOIN rocpd_string s ON o.description_id = s.id"
        ).fetchone()
        assert row is not None
        assert row[0] == ""
        conn.close()

    def test_long_message_roctx(self, tmp_rpd):
        """1000+ character message should be stored and retrieved correctly."""
        long_msg = "A" * 1500
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        insert_roctx_record(conn, long_msg, 1000000, 5000)
        conn.commit()

        row = conn.execute(
            "SELECT s.string FROM rocpd_op o "
            "JOIN rocpd_string s ON o.description_id = s.id"
        ).fetchone()
        assert row is not None
        assert row[0] == long_msg
        assert len(row[0]) == 1500
        conn.close()

    def test_roctx_not_in_top_view(self, tmp_rpd):
        """Roctx markers (gpuId=-1) must not appear in the top view."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        insert_kernel_record(conn, "real_kernel.kd", 0, 1000000, 5000)
        insert_roctx_record(conn, "user_marker", 1000000, 50000)
        conn.commit()

        names = [r[0] for r in conn.execute("SELECT Name FROM top").fetchall()]
        assert "real_kernel.kd" in names
        assert "user_marker" not in names
        conn.close()

    def test_roctx_and_kernels_coexist(self, tmp_rpd):
        """Roctx and kernel ops coexist with correct counts."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)

        for i in range(3):
            insert_roctx_record(conn, "marker_{}".format(i), 1000000 + i * 100000, 5000)
        for i in range(5):
            insert_kernel_record(conn, "kern_{}".format(i), 0, 2000000 + i * 100000, 3000)
        conn.commit()

        total = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        assert total == 8

        roctx_count = conn.execute(
            "SELECT count(*) FROM rocpd_op WHERE gpuId = -1"
        ).fetchone()[0]
        assert roctx_count == 3

        kernel_count = conn.execute(
            "SELECT count(*) FROM rocpd_op WHERE gpuId >= 0"
        ).fetchone()[0]
        assert kernel_count == 5

        top_names = [r[0] for r in conn.execute("SELECT Name FROM top").fetchall()]
        assert len(top_names) == 5
        for name in top_names:
            assert name.startswith("kern_")
        conn.close()


# ---------------------------------------------------------------------------
# Symbol Export Guard Tests
# ---------------------------------------------------------------------------

class TestSymbolExportGuard:

    @SKIP_NO_SO
    def test_roctx_symbols_exported(self):
        """All roctx API symbols must be exported from librtl.so."""
        required = [
            "roctxRangePushA",
            "roctxRangePop",
            "roctxMarkA",
            "roctxRangeStartA",
            "roctxRangeStop",
            "roctxMark",
            "roctxRangePush",
        ]
        result = subprocess.run(
            ["nm", "-D", LIBRTL_PATH],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        symbols = result.stdout.decode()
        for sym in required:
            assert sym in symbols, "Missing exported symbol: {}".format(sym)

    @SKIP_NO_SO
    def test_onload_onunload_exported(self):
        """OnLoad and OnUnload must be exported."""
        result = subprocess.run(
            ["nm", "-D", LIBRTL_PATH],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        symbols = result.stdout.decode()
        assert "OnLoad" in symbols, "Missing OnLoad symbol"
        assert "OnUnload" in symbols, "Missing OnUnload symbol"

    @SKIP_NO_SO
    def test_no_roctracer_rocprofiler_symbols(self):
        """librtl.so must not export roctracer_* or rocprofiler_* symbols."""
        import re as _re
        result = subprocess.run(
            ["nm", "-D", LIBRTL_PATH],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        symbols = result.stdout.decode()
        roctracer_syms = _re.findall(r"\broctracer_\w+", symbols)
        rocprofiler_syms = _re.findall(r"\brocprofiler_\w+", symbols)
        assert not roctracer_syms, "Found roctracer symbols: {}".format(roctracer_syms)
        assert not rocprofiler_syms, "Found rocprofiler symbols: {}".format(rocprofiler_syms)


# ---------------------------------------------------------------------------
# rpd2trace.py vs cmd_convert.py Consistency
# ---------------------------------------------------------------------------

class TestConverterConsistency:

    def test_same_input_same_output(self, tmp_rpd, tmp_path):
        """rpd2trace.py and cmd_convert must produce identical JSON for the same trace."""
        populate_synthetic_trace(tmp_rpd, num_kernels=20)

        # rpd2trace.py via subprocess
        out_rpd2trace = str(tmp_path / "rpd2trace_out.json")
        subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_rpd2trace],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # cmd_convert via import
        out_cmd = str(tmp_path / "cmd_convert_out.json")
        from rocm_trace_lite.cmd_convert import convert
        convert(tmp_rpd, out_cmd)

        with open(out_rpd2trace) as f:
            data1 = json.load(f)
        with open(out_cmd) as f:
            data2 = json.load(f)

        assert data1 == data2

    def test_empty_trace_no_crash_both(self, tmp_rpd, tmp_path):
        """Empty trace must not crash either converter."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        conn.close()

        # rpd2trace.py
        out1 = str(tmp_path / "rpd2trace_empty.json")
        result = subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out1],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert result.returncode == 0

        # cmd_convert -- should not raise
        out2 = str(tmp_path / "cmd_convert_empty.json")
        from rocm_trace_lite.cmd_convert import convert
        convert(tmp_rpd, out2)
        # Neither should crash; output file may or may not be created


# ---------------------------------------------------------------------------
# Kernel Name Fallback
# ---------------------------------------------------------------------------

class TestKernelNameFallback:

    def test_normal_kernel_name_in_top(self, tmp_rpd):
        """Normal kernel name should appear correctly in top view."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        insert_kernel_record(conn, "my_gemm_kernel.kd", 0, 1000000, 5000)
        conn.commit()

        names = [r[0] for r in conn.execute("SELECT Name FROM top").fetchall()]
        assert "my_gemm_kernel.kd" in names
        conn.close()

    def test_hex_fallback_kernel_in_top(self, tmp_rpd):
        """Hex-address kernel name like kernel_0xDEADBEEF should display in top view."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        insert_kernel_record(conn, "kernel_0xDEADBEEF", 0, 1000000, 5000)
        conn.commit()

        names = [r[0] for r in conn.execute("SELECT Name FROM top").fetchall()]
        assert "kernel_0xDEADBEEF" in names
        conn.close()
