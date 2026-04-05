"""Tests for the single-completion-worker architecture (issue #2).

Validates the design invariants introduced by the thread-per-dispatch fix.
Non-GPU tests validate the architecture via source inspection and schema behavior.
"""
import os
import re
import sqlite3
from conftest import populate_synthetic_trace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HSA_INTERCEPT = os.path.join(REPO_ROOT, "src", "hsa_intercept.cpp")


class TestNoDetachedThreads:
    """Verify thread-per-dispatch is gone."""

    def test_no_thread_detach_in_source(self):
        """No std::thread().detach() anywhere in hsa_intercept.cpp."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        assert ".detach()" not in content, (
            "Found .detach() — thread-per-dispatch pattern should be removed"
        )

    def test_no_thread_create_in_intercept_cb(self):
        """queue_intercept_cb should not create threads."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        # Find the queue_intercept_cb function body
        match = re.search(
            r'static void queue_intercept_cb\(.*?\n\}',
            content, re.DOTALL
        )
        assert match, "Could not find queue_intercept_cb"
        cb_body = match.group()
        assert "std::thread" not in cb_body, (
            "queue_intercept_cb creates threads — should enqueue to worker instead"
        )

    def test_worker_thread_exists(self):
        """A single worker thread should be declared."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        assert "g_worker" in content, "No g_worker thread found"
        assert "completion_worker" in content, "No completion_worker function found"

    def test_work_queue_exists(self):
        """A work queue with condition variable should exist."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        assert "g_work_queue" in content, "No work queue found"
        assert "g_work_cv" in content, "No condition variable found"
        assert "g_work_mutex" in content, "No work mutex found"


class TestShutdownCoordination:
    """Verify clean shutdown is implemented."""

    def test_shutdown_flag_exists(self):
        """A global shutdown flag should exist."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        assert "g_shutdown" in content, "No shutdown flag found"

    def test_worker_joins_on_shutdown(self):
        """Worker thread should be joined, not detached."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        assert "g_worker.join()" in content or "g_worker.joinable()" in content, (
            "Worker thread is not joined on shutdown"
        )

    def test_shutdown_called_from_onunload(self):
        """OnUnload should trigger shutdown."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        # Find OnUnload function
        match = re.search(r'extern "C" void OnUnload\(\).*?\n\}', content, re.DOTALL)
        assert match, "Could not find OnUnload"
        assert "shutdown()" in match.group(), "OnUnload does not call shutdown()"

    def test_db_close_after_worker_join(self):
        """DB should close after worker join, not before."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        # Find shutdown function
        match = re.search(r'static void shutdown\(\).*?\n\}', content, re.DOTALL)
        assert match, "Could not find shutdown function"
        body = match.group()
        join_pos = body.find("join()")
        close_pos = body.find("close()")
        assert join_pos >= 0, "No join() in shutdown"
        assert close_pos >= 0, "No close() in shutdown"
        assert join_pos < close_pos, (
            "DB close() happens before worker join() — data loss risk"
        )


class TestBarrierSignalForwarding:
    """Verify barrier-based signal forwarding (not manual load+store)."""

    def test_no_manual_signal_store_for_completion(self):
        """Should not manually store to original_signal in completion handler."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        # Find completion_worker function
        match = re.search(r'static void completion_worker\(\).*?\n\}', content, re.DOTALL)
        assert match, "Could not find completion_worker"
        worker_body = match.group()
        # Should not contain manual signal store for original signal
        assert "original_signal" not in worker_body, (
            "completion_worker accesses original_signal — should be handled by barrier packet"
        )

    def test_barrier_submit_function_exists(self):
        """A barrier submission function should exist."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        assert "submit_barrier_to_forward_signal" in content or "submit_to_signal_queue" in content, (
            "No barrier signal forwarding function found"
        )

    def test_barrier_header_constants(self):
        """HSA barrier packet header constants should be defined."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        assert "kBarrierHeader" in content, "No barrier header constant"
        assert "HSA_PACKET_TYPE_BARRIER_AND" in content, "Not using barrier-and packet type"


class TestQueueInfoTracking:
    """Verify proper queue info lifecycle (fixes memory leak #5)."""

    def test_queue_info_struct_exists(self):
        """QueueInfo struct should exist with device_id and queue_handle."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        assert "struct QueueInfo" in content, "No QueueInfo struct"
        assert "device_id" in content
        assert "queue_handle" in content

    def test_queue_map_exists(self):
        """A map tracking queue info should exist."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        assert "g_queue_map" in content, "No queue map for lifecycle tracking"

    def test_no_raw_new_int_for_device_id(self):
        """Should not use 'new int' for device_id — use QueueInfo instead."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        # Look for the old pattern
        assert "new int(0)" not in content and "new int(" not in content, (
            "Still using raw 'new int' for device_id — should use QueueInfo"
        )

    def test_queue_cleanup_on_shutdown(self):
        """Queue map should be cleared on shutdown."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        assert "g_queue_map.clear()" in content, (
            "Queue map not cleared on shutdown — memory leak"
        )


class TestInterceptCbNonBlocking:
    """Verify the intercept callback is non-blocking."""

    def test_no_signal_wait_in_intercept_cb(self):
        """queue_intercept_cb should never block on signal wait."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        match = re.search(
            r'static void queue_intercept_cb\(.*?\n\}',
            content, re.DOTALL
        )
        assert match
        cb_body = match.group()
        assert "signal_wait" not in cb_body, (
            "queue_intercept_cb blocks on signal wait — should be non-blocking"
        )

    def test_intercept_enqueues_to_worker(self):
        """queue_intercept_cb should push to work queue."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        match = re.search(
            r'static void queue_intercept_cb\(.*?\n\}',
            content, re.DOTALL
        )
        assert match
        cb_body = match.group()
        assert "g_work_queue" in cb_body, (
            "queue_intercept_cb does not enqueue to work queue"
        )
        assert "notify_one" in cb_body or "notify_all" in cb_body, (
            "queue_intercept_cb does not notify worker"
        )


class TestHSAReturnChecks:
    """Verify HSA return values are checked (partial fix for #6)."""

    def test_symbol_info_checks_returns(self):
        """symbol_iterate_cb should check hsa_executable_symbol_get_info returns."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        match = re.search(
            r'static hsa_status_t symbol_iterate_cb\(.*?\n\}',
            content, re.DOTALL
        )
        assert match, "Could not find symbol_iterate_cb"
        body = match.group()
        # Should have return value checks
        assert "!= HSA_STATUS_SUCCESS" in body, (
            "symbol_iterate_cb does not check HSA return values"
        )

    def test_profiling_time_check(self):
        """Dispatch time retrieval should check return status."""
        with open(HSA_INTERCEPT) as f:
            content = f.read()
        match = re.search(
            r'static void completion_worker\(\).*?\n\}',
            content, re.DOTALL
        )
        assert match
        body = match.group()
        assert "HSA_STATUS_SUCCESS" in body, (
            "completion_worker does not check profiling timestamp retrieval status"
        )


class TestHighVolumeSchema:
    """Verify schema handles high-volume workloads correctly."""

    def test_100k_records_no_corruption(self, tmp_rpd):
        """100K synthetic records should all be present and queryable."""
        populate_synthetic_trace(tmp_rpd, num_kernels=100000)
        conn = sqlite3.connect(tmp_rpd)
        count = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        assert count == 100000, f"Expected 100000, got {count}"
        # top view should still work
        top = conn.execute("SELECT count(*) FROM top").fetchone()[0]
        assert top == 5  # 5 unique kernel names
        conn.close()
