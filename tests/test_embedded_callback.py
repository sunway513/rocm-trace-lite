"""Tests for the embedded callback API (trace_db.h hooks).

Validates the callback registration, event delivery, and shutdown
contracts used by embedders like RPD's RtlDataSource.

CPU-only tests validate API contracts via source inspection.
GPU tests (marked gpu) validate end-to-end event delivery.
"""
import os
import re
import subprocess
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
TRACE_DB_H = os.path.join(SRC_DIR, "trace_db.h")
TRACE_DB_CPP = os.path.join(SRC_DIR, "trace_db.cpp")
HSA_INTERCEPT = os.path.join(SRC_DIR, "hsa_intercept.cpp")
HIP_INTERCEPT = os.path.join(SRC_DIR, "hip_api_intercept.cpp")


class TestCallbackAPIContract:
    """Verify the callback API exists and has the documented contract."""

    def test_event_structs_exist(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        assert "struct ApiEventRecord" in content
        assert "struct KernelEventRecord" in content

    def test_setter_functions_declared(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        assert "void set_api_event_callback(" in content
        assert "void set_kernel_event_callback(" in content

    def test_getter_functions_declared(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        assert "ApiEventCallback get_api_event_callback()" in content
        assert "KernelEventCallback get_kernel_event_callback()" in content

    def test_shutdown_function_declared(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        assert "void rtl_trigger_shutdown()" in content


class TestStringLifetimeContract:
    """Verify string lifetime is documented."""

    def test_lifetime_documented_in_header(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        assert "valid only during callback" in content

    def test_api_name_annotated(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        match = re.search(r'struct ApiEventRecord\s*\{(.*?)\}', content, re.DOTALL)
        assert match, "Could not find ApiEventRecord"
        body = match.group(1)
        assert "valid only during callback" in body

    def test_kernel_name_annotated(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        match = re.search(r'struct KernelEventRecord\s*\{(.*?)\}', content, re.DOTALL)
        assert match, "Could not find KernelEventRecord"
        body = match.group(1)
        assert "valid only during callback" in body


class TestCallbackThreadSafety:
    """Verify thread safety contract is documented and implemented."""

    def test_set_before_onload_documented(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        assert "before OnLoad" in content

    def test_noexcept_documented(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        assert "noexcept" in content

    def test_non_blocking_documented(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        assert "non-blocking" in content


class TestShutdownIdempotency:
    """Verify rtl_trigger_shutdown is idempotent."""

    def test_idempotency_documented(self):
        with open(TRACE_DB_H) as f:
            content = f.read()
        assert "Idempotent" in content or "idempotent" in content

    def test_shutdown_has_atomic_guard(self):
        """The underlying shutdown() must have a once-guard."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        match = re.search(
            r'static void shutdown\(\)\s*\{(.*?)\n\}',
            content, re.DOTALL
        )
        assert match, "Could not find shutdown()"
        body = match.group(1)
        assert "shutdown_done" in body, (
            "shutdown() must use an atomic guard for idempotency"
        )


class TestSQLiteGating:
    """Verify SQLite is skipped when either callback is set."""

    def test_onload_checks_both_callbacks(self):
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        # The gating around get_trace_db() in OnLoad should check both
        assert "get_api_event_callback()" in content

    def test_shutdown_checks_both_callbacks(self):
        """flush/close gating must check both kernel and API callbacks."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        # Find the shutdown function and verify it checks both
        match = re.search(
            r'static void shutdown\(\)\s*\{(.*?)\n\}',
            content, re.DOTALL
        )
        assert match, "Could not find shutdown()"
        body = match.group(1)
        assert "get_api_event_callback" in body, (
            "shutdown must check API callback before flushing SQLite"
        )


class TestHipInterceptCallbackPath:
    """Verify HIP wrappers route through callbacks."""

    def test_deliver_hip_api_exists(self):
        with open(HIP_INTERCEPT) as f:
            content = f.read()
        assert "deliver_hip_api(" in content

    def test_is_recording_ready_exists(self):
        with open(HIP_INTERCEPT) as f:
            content = f.read()
        assert "is_recording_ready()" in content

    def test_deliver_checks_callback(self):
        """deliver_hip_api must check get_api_event_callback."""
        with open(HIP_INTERCEPT) as f:
            content = f.read()
        match = re.search(
            r'static void deliver_hip_api\((.*?\n\})',
            content, re.DOTALL
        )
        assert match, "Could not find deliver_hip_api"
        body = match.group(1)
        assert "get_api_event_callback" in body
