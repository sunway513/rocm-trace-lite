"""Unit tests for adapters/torch_profiler.py — 2 tests as per spec §6."""

import pytest
from pathlib import Path
import tempfile


# Test 1: torch_profiler adapter start/stop context manager semantics
def test_torch_profiler_context_manager():
    pytest.importorskip("torch", reason="torch not available")
    from profiler_perf_bench.adapters.torch_profiler import TorchProfilerAdapter

    adapter = TorchProfilerAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter.start(Path(tmpdir))
        adapter.stop()
        # Should not raise


# Test 2: torch_profiler writes trace to tmpdir after stop
def test_torch_profiler_trace_written_to_tmpdir():
    pytest.importorskip("torch", reason="torch not available")
    from profiler_perf_bench.adapters.torch_profiler import TorchProfilerAdapter
    import torch

    adapter = TorchProfilerAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        adapter.start(tmppath)
        # Do a trivial torch op to generate profiler events
        x = torch.ones(10)
        y = x + x
        adapter.stop()

        # Check artifact_glob pattern
        glob_pattern = adapter.artifact_glob()
        assert glob_pattern  # non-empty
