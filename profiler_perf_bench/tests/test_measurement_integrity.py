import json
import os
import sys
import subprocess
import pytest
from profiler_perf_bench.runner import _run_measured, BenchmarkRunner
from profiler_perf_bench.adapters.torch_profiler import TorchProfilerAdapter


def test_rss_does_not_inherit_previous_child_peak(tmp_path):
    if not hasattr(os, 'wait4'):
        pytest.skip('wait4 unavailable')
    def measure(mb):
        cmd = [sys.executable, '-c', f'x=bytearray({mb}*1024*1024); print(len(x))']
        result, rss = _run_measured(cmd, dict(os.environ), str(tmp_path))
        assert result.returncode == 0
        return rss
    high, low = measure(128), measure(1)
    assert high > low + 64, (high, low)


def test_torch_rejects_native_binary(tmp_path):
    with pytest.raises(ValueError, match='Python workload'):
        TorchProfilerAdapter().prepare_run(['/bin/true'], {}, tmp_path)


def test_unbootstrapped_parent_profiler_is_not_reported_as_success(tmp_path):
    from profiler_perf_bench.adapters.base import ExecutionModel
    from profiler_perf_bench.workloads.base import Level
    class Workload:
        name = 'native'
        level = Level.L1
        def cmd(self):
            return ['/bin/true']
        def env(self):
            return {}
    adapter = TorchProfilerAdapter()
    adapter.execution_model = ExecutionModel.IN_PROCESS_PYTHON
    result = BenchmarkRunner(adapter, Workload())._run_once_in(tmp_path)
    assert not result.run_succeeded
    assert result.dropped_reason == 'in_process_adapter_requires_workload_bootstrap'


@pytest.mark.parametrize('exit_code', [0, 7])
def test_child_exit_status_and_trace_survive_system_exit(tmp_path, exit_code):
    pytest.importorskip('torch')
    adapter = TorchProfilerAdapter()
    code = "import torch,sys; x=torch.ones(2); sys.exit(" + str(exit_code) + ")"
    cmd, env = adapter.prepare_run([sys.executable, '-c', code], dict(os.environ), tmp_path)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == exit_code, result.stderr
    assert list(tmp_path.glob(adapter.artifact_glob()))


def test_torch_profiles_workload_process(tmp_path):
    pytest.importorskip('torch')
    import torch
    workload = tmp_path / 'workload.py'
    workload.write_text("import torch\nwith torch.profiler.record_function('target_child_marker'):\n"
                        "    x=torch.ones(32,device='cuda' if torch.cuda.is_available() else 'cpu')\n"
                        "    y=x+x\n"
                        "    if x.is_cuda: torch.cuda.synchronize()\n")
    adapter = TorchProfilerAdapter()
    from profiler_perf_bench.workloads.base import Level
    class Workload:
        name = 'marker-workload'
        level = Level.L2
        def cmd(self):
            return [sys.executable, str(workload)]
        def env(self):
            return {}
        def parse_metrics(self, stdout, stderr, artifact_dir):
            return {}
    result = BenchmarkRunner(adapter, Workload())._run_once_in(tmp_path)
    assert result.run_succeeded, result.dropped_reason
    files = list(tmp_path.glob(adapter.artifact_glob()))
    assert files
    events = json.loads(files[0].read_text())['traceEvents']
    markers = [e for e in events if e.get('name') == 'target_child_marker']
    assert markers and markers[0]['pid'] != os.getpid()
    assert any(e.get('name') == 'aten::add' for e in events)
    if torch.cuda.is_available():
        assert any(e.get('cat') == 'kernel' for e in events), 'No GPU kernel captured'
