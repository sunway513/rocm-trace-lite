"""Unit tests for adapters/base.py — 4 tests as per spec §6."""

import pytest
from profiler_perf_bench.adapters.base import ExecutionModel, ProfilerAdapter


# Test 1: ExecutionModel enum is exhaustive (exactly 2 values as spec §3.2)
def test_execution_model_enum_exhaustive():
    values = {e.value for e in ExecutionModel}
    assert values == {"external_wrapper", "in_process_python"}


# Test 2: ProfilerAdapter is abstract — cannot instantiate directly
def test_profiler_adapter_is_abstract():
    with pytest.raises(TypeError):
        ProfilerAdapter()  # type: ignore


# Test 3: Concrete subclass without required abstract methods raises TypeError
def test_profiler_adapter_requires_abstract_methods():
    class IncompleteAdapter(ProfilerAdapter):
        pass

    with pytest.raises(TypeError):
        IncompleteAdapter()


# Test 4: Concrete subclass with all methods can be instantiated and has required attributes
def test_profiler_adapter_concrete_subclass():
    from pathlib import Path

    class ConcreteAdapter(ProfilerAdapter):
        name = "test_adapter"
        execution_model = ExecutionModel.EXTERNAL_WRAPPER

        def prepare_run(self, cmd, env, tmpdir):
            return cmd, env

        def start(self, tmpdir):
            pass

        def stop(self):
            pass

        def artifact_glob(self):
            return "*.trace"

        def config_hash(self):
            return "abc123"

    adapter = ConcreteAdapter()
    assert adapter.name == "test_adapter"
    assert adapter.execution_model == ExecutionModel.EXTERNAL_WRAPPER
    assert adapter.artifact_glob() == "*.trace"
    assert adapter.config_hash() == "abc123"
