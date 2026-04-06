"""Boundary condition and edge-case tests for rocm-trace-lite.

All tests are CPU-only. No GPU required.
"""
import json
import os
import sqlite3
import subprocess
import sys
from unittest.mock import patch

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))
from conftest import SCHEMA_SQL  # noqa: E402


def _run_cli(*args):
    r = subprocess.run(
        [sys.executable, "-m", "rocm_trace_lite.cli"] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=REPO_ROOT,
    )
    return r.returncode, r.stdout.decode(), r.stderr.decode()


# ---------------------------------------------------------------------------
# CLI Boundary Tests
# ---------------------------------------------------------------------------

class TestCLIBoundary:

    def test_unknown_subcommand(self):
        rc, out, err = _run_cli("badcmd")
        assert rc != 0

    def test_no_args_prints_help(self):
        rc, out, err = _run_cli()
        assert rc != 0
        assert "trace" in out or "usage" in out.lower()

    def test_trace_no_command(self):
        rc, out, err = _run_cli("trace")
        assert rc != 0
        assert "no command" in err.lower() or "error" in err.lower()


# ---------------------------------------------------------------------------
# get_lib_path() Tests
# ---------------------------------------------------------------------------

class TestGetLibPath:

    def test_package_local_lib(self, tmp_path):
        fake_pkg = str(tmp_path / "pkg")
        os.makedirs(os.path.join(fake_pkg, "lib"))
        so_path = os.path.join(fake_pkg, "lib", "librtl.so")
        with open(so_path, "w") as f:
            f.write("")

        original_abspath = os.path.abspath

        def fake_abspath(p):
            if p.endswith("__init__.py") or p == os.path.join(fake_pkg, "__init__.py"):
                return os.path.join(fake_pkg, "__init__.py")
            return original_abspath(p)

        # get_lib_path calls os.path.abspath(__file__) then os.path.dirname
        # We need __file__ resolution to point to fake_pkg
        with patch("os.path.abspath", side_effect=fake_abspath):
            from rocm_trace_lite import get_lib_path
            result = get_lib_path()
        assert result == so_path

    def test_fallback_usr_local_lib(self, tmp_path):
        fake_pkg = str(tmp_path / "pkg")
        os.makedirs(fake_pkg)

        original_isfile = os.path.isfile
        original_abspath = os.path.abspath

        def fake_abspath(p):
            if p.endswith("__init__.py"):
                return os.path.join(fake_pkg, "__init__.py")
            return original_abspath(p)

        def fake_isfile(p):
            if p == os.path.join(fake_pkg, "lib", "librtl.so"):
                return False
            if p == "/usr/local/lib/librtl.so":
                return True
            return original_isfile(p)

        with patch("os.path.abspath", side_effect=fake_abspath):
            with patch("os.path.isfile", side_effect=fake_isfile):
                from rocm_trace_lite import get_lib_path
                result = get_lib_path()
        assert result == "/usr/local/lib/librtl.so"

    def test_not_found_raises(self, tmp_path):
        fake_pkg = str(tmp_path / "pkg")
        os.makedirs(fake_pkg)

        original_abspath = os.path.abspath

        def fake_abspath(p):
            if p.endswith("__init__.py"):
                return os.path.join(fake_pkg, "__init__.py")
            return original_abspath(p)

        with patch("os.path.abspath", side_effect=fake_abspath):
            with patch("os.path.isfile", return_value=False):
                from rocm_trace_lite import get_lib_path
                with pytest.raises(FileNotFoundError):
                    get_lib_path()


# ---------------------------------------------------------------------------
# cmd_info Edge Cases
# ---------------------------------------------------------------------------

class TestInfoEdge:

    def test_empty_db(self, tmp_path):
        db = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db)
        conn.executescript(SCHEMA_SQL)
        conn.close()
        rc, out, err = _run_cli("info", db)
        assert rc == 0
        assert "rocpd_op: 0 rows" in out

    def test_missing_file(self):
        rc, out, err = _run_cli("info", "/tmp/_rtl_nonexistent_boundary_test.db")
        assert rc != 0


# ---------------------------------------------------------------------------
# cmd_summary Edge Cases
# ---------------------------------------------------------------------------

class TestSummaryEdge:

    def test_empty_trace(self, tmp_path):
        db = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db)
        conn.executescript(SCHEMA_SQL)
        conn.close()
        rc, out, err = _run_cli("summary", db)
        assert rc == 0
        assert "0" in out

    def test_missing_file(self):
        rc, out, err = _run_cli("summary", "/tmp/_rtl_nonexistent_boundary_test.db")
        assert rc != 0


# ---------------------------------------------------------------------------
# record_copy Synthetic Tests
# ---------------------------------------------------------------------------

class TestRecordCopy:

    def _make_copy_db(self, path):
        conn = sqlite3.connect(path)
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT INTO rocpd_string(id, string) VALUES(1, 'CopyDeviceToDevice')"
        )
        conn.execute(
            "INSERT INTO rocpd_string(id, string) VALUES(2, 'memcpy_H2D')"
        )
        conn.execute(
            "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, "
            "description_id, opType_id) VALUES(0, 0, 0, 1000, 2000, 2, 1)"
        )
        conn.commit()
        return conn

    def test_copy_record_queryable(self, tmp_path):
        db = str(tmp_path / "copy.db")
        conn = self._make_copy_db(db)
        rows = conn.execute(
            "SELECT s.string FROM rocpd_op o "
            "JOIN rocpd_string s ON o.opType_id = s.id"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == "CopyDeviceToDevice"

    def test_copy_not_in_kernel_top_view(self, tmp_path):
        db = str(tmp_path / "copy.db")
        conn = self._make_copy_db(db)
        conn.execute(
            "INSERT INTO rocpd_string(id, string) VALUES(3, 'KernelExecution')"
        )
        conn.execute(
            "INSERT INTO rocpd_string(id, string) VALUES(4, 'my_kernel.kd')"
        )
        conn.execute(
            "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, "
            "description_id, opType_id) VALUES(0, 0, 0, 3000, 5000, 4, 3)"
        )
        conn.commit()
        top_rows = conn.execute("SELECT Name FROM top").fetchall()
        names = [r[0] for r in top_rows]
        conn.close()
        assert "my_kernel.kd" in names


# ---------------------------------------------------------------------------
# HIP API Converter Path Tests
# ---------------------------------------------------------------------------

class TestHIPAPIConvert:

    def _make_api_db(self, path, add_ops=False, add_apis=False):
        conn = sqlite3.connect(path)
        conn.executescript(SCHEMA_SQL)

        sid = 1
        if add_ops:
            conn.execute(
                "INSERT INTO rocpd_string(id, string) VALUES(?, 'KernelExecution')",
                (sid,),
            )
            sid += 1
            conn.execute(
                "INSERT INTO rocpd_string(id, string) VALUES(?, 'gemm_kernel.kd')",
                (sid,),
            )
            desc_id = sid
            sid += 1
            conn.execute(
                "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, "
                "description_id, opType_id) VALUES(0, 0, 0, 1000, 2000, ?, 1)",
                (desc_id,),
            )

        if add_apis:
            conn.execute(
                "INSERT INTO rocpd_string(id, string) VALUES(?, 'hipLaunchKernel')",
                (sid,),
            )
            api_name_id = sid
            sid += 1
            conn.execute(
                "INSERT INTO rocpd_string(id, string) VALUES(?, 'args=...')",
                (sid,),
            )
            args_id = sid
            sid += 1
            conn.execute(
                "INSERT INTO rocpd_api(pid, tid, start, end, apiName_id, args_id) "
                "VALUES(1234, 5678, 500, 900, ?, ?)",
                (api_name_id, args_id),
            )

        conn.commit()
        conn.close()

    def test_api_records_in_json(self, tmp_path):
        db = str(tmp_path / "api.db")
        json_out = str(tmp_path / "api.json")
        self._make_api_db(db, add_ops=True, add_apis=True)
        rc, out, err = _run_cli("convert", db, "-o", json_out)
        assert rc == 0
        with open(json_out) as f:
            trace = json.load(f)
        cats = [e.get("cat") for e in trace["traceEvents"]]
        assert "hip_api" in cats

    def test_empty_api_no_crash(self, tmp_path):
        db = str(tmp_path / "noapi.db")
        json_out = str(tmp_path / "noapi.json")
        self._make_api_db(db, add_ops=True, add_apis=False)
        rc, out, err = _run_cli("convert", db, "-o", json_out)
        assert rc == 0
        with open(json_out) as f:
            trace = json.load(f)
        cats = [e.get("cat") for e in trace["traceEvents"]]
        assert "hip_api" not in cats

    def test_mixed_ops_and_api(self, tmp_path):
        db = str(tmp_path / "mixed.db")
        json_out = str(tmp_path / "mixed.json")
        self._make_api_db(db, add_ops=True, add_apis=True)
        rc, out, err = _run_cli("convert", db, "-o", json_out)
        assert rc == 0
        with open(json_out) as f:
            trace = json.load(f)
        cats = [e.get("cat") for e in trace["traceEvents"]]
        assert "hip_api" in cats
        assert "KernelExecution" in cats
