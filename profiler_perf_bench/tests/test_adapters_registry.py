"""Unit tests for adapters/registry.py — 2 tests as per spec §6."""

import pytest
from profiler_perf_bench.adapters.registry import AdapterRegistry, register_adapter
from profiler_perf_bench.adapters.base import ExecutionModel, ProfilerAdapter


def _make_adapter(name):
    """Helper to create a minimal concrete adapter with given name."""
    class TestAdapter(ProfilerAdapter):
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
            return "hash"

    TestAdapter.name = name
    TestAdapter.__name__ = f"TestAdapter_{name}"
    return TestAdapter


# Test 1: @register_adapter decorator registers adapter by name
def test_register_adapter_decorator():
    registry = AdapterRegistry()

    AdapterCls = _make_adapter("my_test_adapter")

    @registry.register
    class DecoratedAdapter(AdapterCls):
        name = "my_test_adapter_decorated"

    assert "my_test_adapter_decorated" in registry.list_names()


# Test 2: registry.enumerate() returns all registered adapters
def test_registry_enumerate():
    registry = AdapterRegistry()

    A1 = _make_adapter("enum_adapter_1")
    A2 = _make_adapter("enum_adapter_2")

    @registry.register
    class Adapter1(A1):
        name = "enum_adapter_1"

    @registry.register
    class Adapter2(A2):
        name = "enum_adapter_2"

    names = registry.list_names()
    assert "enum_adapter_1" in names
    assert "enum_adapter_2" in names

    all_adapters = registry.enumerate()
    assert len(all_adapters) >= 2


# Test 3 (bonus): duplicate name raises
def test_registry_name_collision():
    registry = AdapterRegistry()

    A = _make_adapter("collision_adapter")

    @registry.register
    class Adapter1(A):
        name = "collision_adapter"

    with pytest.raises(ValueError, match="collision"):
        @registry.register
        class Adapter2(A):
            name = "collision_adapter"
