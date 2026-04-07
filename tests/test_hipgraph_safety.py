"""Tests for HIP Graph safety (issue #15, ADR-001).

Validates timeout-based signal wait and clean shutdown via source code inspection.
GPU graph tests are in test_gpu_hip.py (HIP-native, no torch).
"""
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HSA_FILE = os.path.join(REPO_ROOT, "src", "hsa_intercept.cpp")


# ---- CPU tests: architecture validation ----

class TestTimeoutBasedWait:
    """Verify signal wait uses bounded timeout, not UINT64_MAX."""

    def _get_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def test_no_uint64_max_in_signal_wait(self):
        """completion_worker must not use UINT64_MAX timeout."""
        src = self._get_source()
        match = re.search(r'static void completion_worker\(\).*?\n\}', src, re.DOTALL)
        assert match, "Could not find completion_worker"
        body = match.group()
        assert "UINT64_MAX" not in body, \
            "completion_worker still uses UINT64_MAX — must use bounded timeout"

    def test_timeout_constant_defined(self):
        """A timeout constant (e.g., WAIT_TIMEOUT_NS) should be defined."""
        src = self._get_source()
        assert "WAIT_TIMEOUT" in src, "No timeout constant found"

    def test_shutdown_check_in_wait_loop(self):
        """Wait loop must check g_shutdown between timeouts."""
        src = self._get_source()
        match = re.search(r'static void completion_worker\(\).*?\n\}', src, re.DOTALL)
        body = match.group()
        assert "g_shutdown" in body, "No shutdown check in completion_worker wait loop"
        # Should be a while loop with timeout + shutdown check
        assert "while" in body, "No while loop for timeout-based wait"

    def test_abandoned_dispatch_handled(self):
        """Dispatches abandoned during shutdown must be deleted, not leaked."""
        src = self._get_source()
        match = re.search(r'static void completion_worker\(\).*?\n\}', src, re.DOTALL)
        body = match.group()
        assert "delete dd" in body, "No delete for abandoned dispatch"
        assert "continue" in body, "No continue after abandoning dispatch"


class TestShutdownSafety:
    """Verify shutdown prevents double-call and drains work queue."""

    def _get_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def test_double_shutdown_prevention(self):
        """shutdown() must guard against being called twice."""
        src = self._get_source()
        match = re.search(r'static void shutdown\(\).*?\n\}', src, re.DOTALL)
        assert match
        body = match.group()
        assert "shutdown_done" in body or "once_flag" in body, \
            "No double-shutdown prevention"

    def test_work_queue_drained(self):
        """shutdown() must drain pending items from work queue."""
        src = self._get_source()
        match = re.search(r'static void shutdown\(\).*?\n\}', src, re.DOTALL)
        body = match.group()
        assert "g_work_queue" in body, "shutdown does not touch work queue"
        assert "delete" in body, "shutdown does not delete pending items"

    def test_join_before_close(self):
        """Worker join must happen before DB close."""
        src = self._get_source()
        match = re.search(r'static void shutdown\(\).*?\n\}', src, re.DOTALL)
        body = match.group()
        join_pos = body.find("join()")
        close_pos = body.find("close()")
        assert join_pos < close_pos, "DB close before worker join"

    def test_adr_document_exists(self):
        """ADR-001 document must exist."""
        adr = os.path.join(REPO_ROOT, "docs", "ADR-001-hipgraph-safety.md")
        assert os.path.exists(adr), "Missing ADR-001 document"


# GPU graph tests moved to test_gpu_hip.py (HIP-native, no torch dependency)
