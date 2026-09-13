"""Validation-only vLLM worker extension: rank markers, graph evidence, flush."""
import ctypes
import json
import os
import time
from pathlib import Path


class TraceValidationWorker:
    def rtl_latency_setup_barrier(self):
        """Drain this worker after warmup, including the legacy default stream."""
        import torch
        torch.cuda.synchronize()
        self._rtl_latency_replays = 0
        if not getattr(self, '_rtl_latency_counter_installed', False):
            original = torch.cuda.CUDAGraph.replay
            def counted_replay(graph, *args, **kwargs):
                self._rtl_latency_replays += 1
                return original(graph, *args, **kwargs)
            torch.cuda.CUDAGraph.replay = counted_replay
            self._rtl_latency_counter_installed = True
        maps = Path('/proc/self/maps').read_text().splitlines()
        libraries = sorted({line.split()[-1] for line in maps
                            if 'librtl.so' in line or 'libamdhip64.so' in line})
        import hashlib
        hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in libraries}
        return {'rank': self.rank, 'pid': os.getpid(),
                'device': torch.cuda.current_device(), 'synchronized': True,
                'monotonic_ns': time.monotonic_ns(), 'loaded_libraries': hashes}

    def rtl_latency_end(self):
        import torch
        torch.cuda.synchronize()
        return {'rank': self.rank, 'pid': os.getpid(),
                'graph_replays': self._rtl_latency_replays,
                'monotonic_ns': time.monotonic_ns()}

    def rtl_e2e_begin(self, output_dir):
        import torch
        torch.cuda.synchronize()
        self._rtl_e2e_dir = Path(output_dir)
        self._rtl_e2e_dir.mkdir(parents=True, exist_ok=True)
        self._rtl_e2e_stats = dict(pid=os.getpid(), rank=self.rank,
                                   device=torch.cuda.current_device(), replays=0)
        lib = ctypes.CDLL(None)
        mark = lib.roctxMarkA
        mark.argtypes = [ctypes.c_char_p]
        original = torch.cuda.CUDAGraph.replay
        stats = self._rtl_e2e_stats

        def replay(graph, *args, **kwargs):
            stats['replays'] += 1
            mark(f"e2e_graph_replay rank={self.rank} index={stats['replays']}".encode())
            return original(graph, *args, **kwargs)

        torch.cuda.CUDAGraph.replay = replay
        mark(f"e2e_requests_begin rank={self.rank}".encode())
        return {**stats, 'monotonic_ns': time.monotonic_ns()}

    def rtl_e2e_end(self):
        import torch
        torch.cuda.synchronize()
        lib = ctypes.CDLL(None)
        lib.roctxMarkA(f"e2e_requests_end rank={self.rank}".encode())
        result = {**self._rtl_e2e_stats, 'monotonic_ns': time.monotonic_ns()}
        (self._rtl_e2e_dir / f'rank-{self.rank}.json').write_text(json.dumps(result, indent=2))
        return result

    def rtl_e2e_finalize(self):
        """Drain after completed requests, before serving worker teardown."""
        import torch
        torch.cuda.synchronize()
        lib = ctypes.CDLL(None)
        shutdown = getattr(lib, '_ZN8trace_db20rtl_trigger_shutdownEv')
        shutdown.argtypes = []
        shutdown.restype = None
        shutdown()
        return {'pid': os.getpid(), 'rank': self.rank, 'flushed': True}
