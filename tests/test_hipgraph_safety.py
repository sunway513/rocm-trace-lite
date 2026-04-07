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


class TestBatchSkip:
    """Verify batch mode (count > 1) skips signal injection (issue #67)."""

    def _get_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def _get_interceptor(self):
        src = self._get_source()
        match = re.search(r'static void queue_intercept_cb\(.*?\n\}', src, re.DOTALL)
        assert match, "Could not find queue_intercept_cb"
        return match.group()

    def test_batch_mode_detected(self):
        """Batch mode (count > 1) must be detected."""
        body = self._get_interceptor()
        assert "batch_mode" in body or "count > 1" in body, \
            "No batch mode detection in interceptor"

    def test_batch_skip_passthrough(self):
        """Batch submissions must be passed through unmodified."""
        body = self._get_interceptor()
        assert "batch_mode" in body, "No batch_mode variable"
        # Must call writer() and return early for batch
        assert "writer(in_packets, count)" in body, \
            "Batch mode must call writer(in_packets, count) to pass through unmodified"

    def test_batch_skip_counter(self):
        """Batch skip must increment a counter for diagnostics."""
        body = self._get_interceptor()
        # Should increment drop counter for batch mode
        assert "g_drop_not_kernel" in body, \
            "No drop counter for batch skip"

    def test_no_signal_injection_in_batch(self):
        """Signal injection must not happen inside batch_mode block."""
        body = self._get_interceptor()
        # The batch_mode block should return early before the signal injection loop
        batch_block = re.search(
            r'if\s*\(batch_mode\)\s*\{.*?\breturn\b', body, re.DOTALL)
        assert batch_block, \
            "batch_mode block must return early before signal injection"


class TestSignalSkip:
    """Verify packets with existing completion_signal are skipped (issue #67)."""

    def _get_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def test_completion_signal_check(self):
        """Packets with non-zero completion_signal must be skipped."""
        src = self._get_source()
        assert "completion_signal.handle != 0" in src, \
            "No check for existing completion_signal"

    def test_skip_has_signal_counter(self):
        """Skipped-signal counter must exist for diagnostics."""
        src = self._get_source()
        assert "g_skip_has_signal" in src, \
            "No g_skip_has_signal counter"

    def test_skip_counter_in_shutdown(self):
        """g_skip_has_signal must be reported in shutdown stats."""
        src = self._get_source()
        match = re.search(r'static void shutdown\(\).*?\n\}', src, re.DOTALL)
        assert match
        body = match.group()
        assert "g_skip_has_signal" in body, \
            "g_skip_has_signal not reported in shutdown"


class TestSafeMode:
    """Verify RTL_SAFE_MODE disables intercept queue (issue #67)."""

    def _get_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def test_safe_mode_env_var(self):
        """RTL_SAFE_MODE env var must be checked."""
        src = self._get_source()
        assert "RTL_SAFE_MODE" in src, \
            "No RTL_SAFE_MODE env var support"

    def test_safe_mode_disables_intercept(self):
        """Safe mode must disable hsa_amd_queue_intercept_create."""
        src = self._get_source()
        assert "g_intercept_available = false" in src or \
               "intercept disabled" in src, \
            "Safe mode does not disable intercept"

    def test_safe_mode_plain_queue_profiling(self):
        """Safe mode must enable profiling on plain queue."""
        src = self._get_source()
        # In the safe_mode/!intercept path, profiling should still be enabled
        match = re.search(
            r'if\s*\(!g_intercept_available\s*\|\|\s*safe_mode\).*?\}',
            src, re.DOTALL)
        assert match, "No safe_mode/!intercept queue path found"
        body = match.group()
        assert "profiling_set_profiler_enabled" in body, \
            "Safe mode must enable profiling on plain queue"


class TestDebugLogging:
    """Verify RTL_DEBUG diagnostic logging (issue #67)."""

    def _get_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def test_debug_env_var(self):
        """RTL_DEBUG env var must be checked."""
        src = self._get_source()
        assert "RTL_DEBUG" in src, "No RTL_DEBUG env var"

    def test_debug_levels(self):
        """Debug level 1 and 2 must be supported."""
        src = self._get_source()
        assert "debug_level >= 1" in src, "No debug level 1"
        assert "debug_level >= 2" in src, "No debug level 2"


# GPU graph tests are in test_gpu_hip.py (HIP-native, no torch dependency)
