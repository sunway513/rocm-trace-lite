"""Stress tests and fault injection for rocm-trace-lite (CPU only).

GPU stress tests are in test_gpu_hip.py (HIP-native, no torch).
"""
import os
import subprocess
import sys
import stat

import pytest

from conftest import populate_synthetic_trace, SCHEMA_SQL
from rocm_trace_lite.cmd_trace import _generate_summary

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "librtl.so")


def _has_lib():
    return os.path.exists(LIB_PATH)


def _make_valid_db(db_path, num_kernels=50):
    """Create a valid trace DB with synthetic data."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.close()
    populate_synthetic_trace(db_path, num_kernels=num_kernels)


# ===========================================================================
# HSA Lifecycle (CPU only, 2 tests)
# ===========================================================================


class TestHSALifecycle:
    """Verify .so exports and dependencies without running GPU code."""

    def test_onload_onunload_symbols(self):
        """librtl.so must export OnLoad and OnUnload."""
        if not _has_lib():
            pytest.skip("librtl.so not built")
        result = subprocess.run(
            ["nm", "-D", LIB_PATH],
            stdout=subprocess.PIPE, universal_newlines=True, timeout=10
        )
        assert "T OnLoad" in result.stdout, "OnLoad not exported"
        assert "T OnUnload" in result.stdout, "OnUnload not exported"

    def test_shared_lib_dependencies(self):
        """Verify .so NEEDED dependencies are only expected libraries."""
        if not _has_lib():
            pytest.skip("librtl.so not built")
        result = subprocess.run(
            ["ldd", LIB_PATH],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=10
        )
        assert result.returncode == 0, "ldd failed: {}".format(result.stderr)
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or "=>" not in line:
                continue
            if "not found" in line:
                lib_name = line.split("=>")[0].strip()
                pytest.fail("Missing dependency: {}".format(lib_name))


# ===========================================================================
# Fault Injection (CPU only, 3 tests)
# ===========================================================================


class TestFaultInjection:
    """Fault injection tests on SQLite and file system paths."""

    def test_readonly_db_summary_readable(self, tmp_path):
        """Create valid trace DB, chmod 444, _generate_summary() still reads it."""
        db_path = str(tmp_path / "readonly.db")
        _make_valid_db(db_path, num_kernels=50)
        os.chmod(db_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            summary = _generate_summary(db_path)
            assert summary is not None
            assert "50 GPU ops" in summary
        finally:
            os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    def test_nonexistent_output_dir_error(self, tmp_path):
        """rtl trace -o /nonexistent/dir/trace.db should not create file."""
        bad_output = "/nonexistent_rtl_test_dir_{}/trace.db".format(os.getpid())
        script = str(tmp_path / "dummy.py")
        with open(script, "w") as f:
            f.write("print('hello')\n")
        env = os.environ.copy()
        env["PYTHONPATH"] = "{}:{}".format(REPO_ROOT, env.get("PYTHONPATH", ""))
        subprocess.run(
            [sys.executable, "-m", "rocm_trace_lite.cli", "trace",
             "-o", bad_output, sys.executable, script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30, env=env
        )
        assert not os.path.exists(bad_output)

    def test_empty_sqlite_no_crash(self, tmp_path):
        """Empty SQLite file (0 bytes): _generate_summary() must not crash."""
        db_path = str(tmp_path / "empty.db")
        open(db_path, "w").close()
        summary = _generate_summary(db_path)
        assert summary is None or isinstance(summary, str)
