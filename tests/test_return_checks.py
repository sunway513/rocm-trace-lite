"""Tests for HSA/SQLite return value checking (issue #6).

Validates that critical API calls check return status and that
the TraceDB tracks dropped records.
"""
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HSA_FILE = os.path.join(REPO_ROOT, "src", "hsa_intercept.cpp")
DB_FILE = os.path.join(REPO_ROOT, "src", "rpd_lite.cpp")
HDR_FILE = os.path.join(REPO_ROOT, "src", "rpd_lite.h")


class TestHSAReturnChecks:
    """HSA API calls must check return status."""

    def _get_hsa_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def test_profiling_get_dispatch_time_checked(self):
        src = self._get_hsa_source()
        match = re.search(
            r'hsa_amd_profiling_get_dispatch_time_fn.*?;(.*?)(?:record_kernel|signal_destroy)',
            src, re.DOTALL
        )
        assert match, "Could not find dispatch_time call context"
        assert "HSA_STATUS_SUCCESS" in match.group(), \
            "profiling_get_dispatch_time return not checked"

    def test_profiling_set_profiler_enabled_checked(self):
        src = self._get_hsa_source()
        # Find the set_profiler_enabled call and check nearby lines
        idx = src.find("hsa_amd_profiling_set_profiler_enabled_fn")
        assert idx >= 0
        context = src[idx:idx+300]
        assert "HSA_STATUS_SUCCESS" in context or "prof_status" in context, \
            "profiling_set_profiler_enabled return not checked"

    def test_signal_destroy_checked(self):
        src = self._get_hsa_source()
        match = re.search(
            r'(hsa_signal_destroy_fn.*?;.*?\n.*?\n.*?\n)',
            src, re.DOTALL
        )
        assert match
        context = match.group()
        assert "status" in context.lower(), \
            "signal_destroy return not checked"

    def test_signal_wait_result_checked(self):
        src = self._get_hsa_source()
        match = re.search(r'completion_worker\(\).*?\n\}', src, re.DOTALL)
        assert match
        body = match.group()
        assert "wait_val" in body or "signal_wait" in body.split("warning")[0], \
            "signal_wait return value not inspected"

    def test_symbol_info_checks(self):
        """Already validated in test_completion_worker, re-check here."""
        src = self._get_hsa_source()
        match = re.search(r'symbol_iterate_cb\(.*?\n\}', src, re.DOTALL)
        assert match
        body = match.group()
        assert body.count("HSA_STATUS_SUCCESS") >= 3, \
            "symbol_iterate_cb should check at least 3 HSA info calls"


class TestSQLiteReturnChecks:
    """SQLite API calls must check return status."""

    def _get_db_source(self):
        with open(DB_FILE) as f:
            return f.read()

    def test_prepare_v2_checked(self):
        src = self._get_db_source()
        assert "SQLITE_OK" in src or "prepare.*failed" in src, \
            "sqlite3_prepare_v2 returns not checked"

    def test_begin_transaction_checked(self):
        src = self._get_db_source()
        match = re.search(r'begin_transaction\(\).*?\n\}', src, re.DOTALL)
        assert match
        body = match.group()
        assert "SQLITE_OK" in body, "BEGIN TRANSACTION return not checked"

    def test_commit_transaction_checked(self):
        src = self._get_db_source()
        match = re.search(r'commit_transaction\(\).*?\n\}', src, re.DOTALL)
        assert match
        body = match.group()
        assert "SQLITE_OK" in body, "COMMIT return not checked"

    def test_step_results_tracked(self):
        """sqlite3_step failures should increment records_dropped_."""
        src = self._get_db_source()
        assert "records_dropped_" in src, "No dropped record tracking"
        assert "records_written_" in src, "No written record tracking"

    def test_step_helper_exists(self):
        """A step_ok helper should wrap sqlite3_step return checking."""
        src = self._get_db_source()
        assert "step_ok" in src, "No step_ok helper found"
        assert "SQLITE_DONE" in src, "step_ok does not check SQLITE_DONE"


class TestDropReporting:
    """Dropped records should be reported at close."""

    def test_close_reports_stats(self):
        with open(DB_FILE) as f:
            src = f.read()
        match = re.search(r'void TraceDB::close\(\).*?\n\}', src, re.DOTALL)
        assert match
        body = match.group()
        assert "records_written_" in body, "close() does not report written count"
        assert "records_dropped_" in body or "DROPPED" in body, \
            "close() does not report dropped count"

    def test_header_has_counters(self):
        with open(HDR_FILE) as f:
            src = f.read()
        assert "records_written_" in src, "Header missing records_written_ field"
        assert "records_dropped_" in src, "Header missing records_dropped_ field"
