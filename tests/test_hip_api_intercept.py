"""CPU unit tests for HIP API interception (no GPU required)."""
import os
import sqlite3
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")


class TestHipApiSchema:
    """Verify DB schema supports HIP API events."""

    def _create_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        # Read schema from trace_db.cpp source
        schema_src = open(os.path.join(SRC_DIR, "trace_db.cpp")).read()
        import re
        m = re.search(r'R"SQL\((.*?)\)SQL"', schema_src, re.DOTALL)
        assert m, "Cannot find schema in trace_db.cpp"
        conn = sqlite3.connect(db_path)
        conn.executescript(m.group(1))
        conn.close()
        return db_path

    def test_rocpd_api_table_exists(self, tmp_path):
        db_path = self._create_db(tmp_path)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "rocpd_api" in tables

    def test_rocpd_api_columns(self, tmp_path):
        db_path = self._create_db(tmp_path)
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(rocpd_api)").fetchall()]
        for expected in ["id", "pid", "tid", "start", "end", "apiName_id", "args_id"]:
            assert expected in cols, f"Missing column: {expected}"

    def test_rocpd_api_ops_table_exists(self, tmp_path):
        db_path = self._create_db(tmp_path)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "rocpd_api_ops" in tables

    def test_rocpd_api_ops_columns(self, tmp_path):
        db_path = self._create_db(tmp_path)
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(rocpd_api_ops)").fetchall()]
        assert "api_id" in cols
        assert "op_id" in cols


class TestHipApiSourceGuard:
    """Verify HIP API intercept source has correct patterns."""

    def test_reentrance_guard_present(self):
        src = open(os.path.join(SRC_DIR, "hip_api_intercept.cpp")).read()
        assert "tls_in_hip_api" in src, "Missing re-entrancy guard"

    def test_dlsym_rtld_next(self):
        src = open(os.path.join(SRC_DIR, "hip_api_intercept.cpp")).read()
        assert "RTLD_NEXT" in src, "Must use RTLD_NEXT for forwarding"

    def test_no_hip_library_link(self):
        makefile = open(os.path.join(REPO_ROOT, "Makefile")).read()
        assert "-lamdhip64" not in makefile, "Must not link against libamdhip64"
        assert "-lhip_hcc" not in makefile, "Must not link against libhip_hcc"

    def test_no_libamdhip64_in_binary(self):
        lib = os.path.join(REPO_ROOT, "librtl.so")
        if not os.path.exists(lib):
            pytest.skip("librtl.so not built")
        result = subprocess.run(["ldd", lib], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        assert "libamdhip64" not in result.stdout, "librtl.so must not depend on libamdhip64"

    def test_correlation_map_present(self):
        src = open(os.path.join(SRC_DIR, "hip_api_intercept.h")).read()
        assert "CorrelationMap" in src
        assert "g_correlation_map" in src

    def test_hip_wrappers_extern_c(self):
        src = open(os.path.join(SRC_DIR, "hip_api_intercept.cpp")).read()
        assert 'extern "C"' in src, "HIP wrappers must be extern C"

    def test_all_wrappers_have_guard(self):
        src = open(os.path.join(SRC_DIR, "hip_api_intercept.cpp")).read()
        wrappers = ["hipModuleLaunchKernel", "hipExtModuleLaunchKernel",
                     "hipMemcpy", "hipMemcpyAsync", "hipMalloc", "hipFree",
                     "hipStreamSynchronize", "hipDeviceSynchronize",
                     "hipGraphLaunch",
                     "hipSetDevice", "hipStreamCreate", "hipStreamDestroy",
                     "hipEventCreate", "hipEventDestroy",
                     "hipEventRecord", "hipEventSynchronize",
                     "hipGraphCreate", "hipGraphInstantiate",
                     "hipGraphExecDestroy",
                     "hipHostMalloc", "hipHostFree"]
        for w in wrappers:
            assert f"resolve_{w}" in src, f"Missing resolve for {w}"

    def test_enabled_flag_default_false(self):
        src = open(os.path.join(SRC_DIR, "hip_api_intercept.cpp")).read()
        assert "g_enabled{false}" in src, "HIP API must be disabled by default"
