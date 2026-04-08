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
        """Batch skip must increment a dedicated counter for diagnostics."""
        body = self._get_interceptor()
        # Should increment dedicated batch skip counter, not g_drop_not_kernel
        assert "g_drop_batch_skip" in body, \
            "No dedicated batch skip counter"

    def test_no_signal_injection_in_batch(self):
        """Batch submissions must return early (no signal injection) in default/lite modes."""
        body = self._get_interceptor()
        # The batch_mode block should return early before the signal injection loop
        # Pattern: if (batch_mode && g_rtl_mode != RtlMode::FULL) { ... return; }
        batch_block = re.search(
            r'if\s*\(batch_mode.*?\{.*?\breturn\b', body, re.DOTALL)
        assert batch_block, \
            "batch_mode block must return early before signal injection"


class TestSignalForwarding:
    """Verify packets with app-provided completion_signal are still profiled."""

    def _get_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def test_original_signal_saved(self):
        """Original completion_signal must be saved before injection."""
        src = self._get_source()
        assert "orig_sig = pkt->completion_signal" in src, \
            "Original signal not saved before injection"

    def test_original_signal_forwarded(self):
        """Original signal must be forwarded after timestamp collection."""
        src = self._get_source()
        assert "original_signal" in src and "subtract_screlease" in src, \
            "Original signal not forwarded via subtract_screlease"


class TestRtlModes:
    """Verify RTL_MODE profiling modes (default/lite/full)."""

    def _get_source(self):
        with open(HSA_FILE) as f:
            return f.read()

    def _get_interceptor(self):
        src = self._get_source()
        match = re.search(r'static void queue_intercept_cb\(.*?\n\}', src, re.DOTALL)
        assert match, "Could not find queue_intercept_cb"
        return match.group()

    def test_rtl_mode_enum_defined(self):
        """RtlMode enum must define DEFAULT, LITE, FULL."""
        src = self._get_source()
        assert "RtlMode" in src, "No RtlMode enum"
        assert "DEFAULT" in src, "No DEFAULT mode"
        assert "LITE" in src, "No LITE mode"
        assert "FULL" in src, "No FULL mode"

    def test_rtl_mode_env_var_parsed(self):
        """RTL_MODE env var must be read in OnLoad."""
        src = self._get_source()
        assert 'getenv("RTL_MODE")' in src, "RTL_MODE env var not read"

    def test_default_mode_is_default(self):
        """Default mode must be DEFAULT (not LITE or FULL)."""
        src = self._get_source()
        assert "g_rtl_mode = RtlMode::DEFAULT" in src, \
            "Default mode is not DEFAULT"

    def test_mode_names_printed(self):
        """Mode name must be printed at startup."""
        src = self._get_source()
        assert 'mode_names' in src and '"default"' in src and '"lite"' in src and '"full"' in src, \
            "Mode names not defined for printing"

    def test_lite_mode_skips_has_signal(self):
        """Lite mode must skip packets with existing completion_signal."""
        body = self._get_interceptor()
        assert "RtlMode::LITE" in body and "completion_signal" in body, \
            "Lite mode does not check completion_signal"

    def test_full_mode_allows_batch(self):
        """Full mode must NOT skip batch submissions (count > 1)."""
        body = self._get_interceptor()
        assert "RtlMode::FULL" in body, \
            "Full mode not referenced in batch skip logic"
        # The batch skip should check g_rtl_mode != FULL
        assert "g_rtl_mode != RtlMode::FULL" in body, \
            "Batch skip does not exempt FULL mode"

    def test_full_mode_rocr_requirement_documented(self):
        """Full mode must document ROCm 7.13+ / ROCR fix requirement."""
        src = self._get_source()
        assert "559d48b1" in src or "ROCm 7.13" in src or "rocm-systems" in src, \
            "Full mode ROCR requirement not documented in source"

    def test_default_skips_graph_replay(self):
        """Default mode must skip graph replay batches (count > 1)."""
        body = self._get_interceptor()
        # batch_mode check should apply when mode is not FULL
        batch_check = re.search(r'if\s*\(batch_mode.*?RtlMode::FULL', body, re.DOTALL)
        assert batch_check, "Default mode batch skip logic not found"

    def test_mode_lite_parse(self):
        """RTL_MODE=lite must set LITE mode."""
        src = self._get_source()
        assert '"lite"' in src, "lite mode string not parsed"
        assert "RtlMode::LITE" in src, "LITE mode not set"

    def test_mode_full_parse(self):
        """RTL_MODE=full must set FULL mode."""
        src = self._get_source()
        assert '"full"' in src, "full mode string not parsed"
        assert "RtlMode::FULL" in src, "FULL mode not set"

    def test_unknown_mode_warning(self):
        """Unknown RTL_MODE value must print a warning."""
        src = self._get_source()
        assert "unknown RTL_MODE" in src or "WARNING" in src, \
            "No warning for unknown RTL_MODE value"


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
