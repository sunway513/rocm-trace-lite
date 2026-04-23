"""Unit tests for workloads/base.py — 4 tests as per spec §6."""

import pytest
from profiler_perf_bench.workloads.base import Level, Workload


# Test 1: Level enum has exactly L1, L2, L3
def test_level_enum_exhaustive():
    levels = {e.name for e in Level}
    assert levels == {"L1", "L2", "L3"}
    assert Level.L1.value == 1
    assert Level.L2.value == 2
    assert Level.L3.value == 3


# Test 2: Workload is abstract — cannot instantiate directly
def test_workload_is_abstract():
    with pytest.raises(TypeError):
        Workload()


# Test 3: Concrete workload satisfies contract — cmd/env/ready_probe
def test_workload_concrete_subclass_contract():
    class ConcreteWorkload(Workload):
        name = "test_workload"
        level = Level.L1
        requires = ["hipcc"]

        def cmd(self):
            return ["./gpu_workload", "gemm", "64", "500"]

        def env(self):
            return {}

        def ready_probe(self):
            return None

        def client_cmd(self):
            return None

        def parse_metrics(self, stdout, stderr, artifact_dir):
            return {"wall_s": 1.0}

    w = ConcreteWorkload()
    assert w.name == "test_workload"
    assert w.level == Level.L1
    assert w.requires == ["hipcc"]
    assert w.cmd() == ["./gpu_workload", "gemm", "64", "500"]
    assert w.env() == {}
    assert w.ready_probe() is None
    assert w.client_cmd() is None


# Test 4: requires-list evaluation helper checks which requirements are satisfied
def test_workload_check_requires():
    class TestWorkload(Workload):
        name = "req_test"
        level = Level.L1
        requires = ["true"]  # 'true' is always available on Linux

        def cmd(self):
            return ["true"]

        def env(self):
            return {}

        def ready_probe(self):
            return None

        def client_cmd(self):
            return None

        def parse_metrics(self, stdout, stderr, artifact_dir):
            return {}

    w = TestWorkload()
    # check_requires() should return list of missing requirements
    missing = w.check_requires()
    # "true" is a real command, so nothing should be missing
    assert "true" not in missing
