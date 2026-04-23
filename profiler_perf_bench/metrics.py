"""Metrics schema and data classes for perf-only benchmarking.

All types are JSON-serializable to enable easy persistence and comparison.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class UniversalMetrics(TypedDict, total=False):
    """Per-run metrics collected for all adapters and workload levels."""
    wall_s: float                   # full end-to-end wall time
    subprocess_s: float             # subprocess.Popen wall time
    adapter_init_s: Optional[float] # time from start to profiler ready; None if unobservable
    adapter_shutdown_s: Optional[float]
    trace_bytes: int                # total bytes of profiler artifacts under tmpdir
    peak_rss_MB: float
    run_succeeded: bool             # True iff sanity gates all pass
    dropped_reason: Optional[str]  # set iff run_succeeded=False


class L3Metrics(TypedDict, total=False):
    """L3-specific LLM serving metrics (optional, populated by L3 workloads)."""
    ttft_ms_mean: float
    ttft_ms_median: float
    ttft_ms_p99: float
    itl_ms_mean: float
    itl_ms_median: float
    itl_ms_p99: float
    tpot_ms_mean: float
    tpot_ms_median: float
    tpot_ms_p99: float
    e2e_latency_ms_mean: float
    output_tokens_per_sec: float
    request_throughput_rps: float
    successful_requests: int
    total_requests: int
    bench_duration_s: float


@dataclass
class RunResult:
    """Result from a single run_once() call."""
    adapter_name: str
    workload_name: str
    round_idx: int
    metrics: Dict[str, Any]   # UniversalMetrics + optional L3Metrics fields
    run_succeeded: bool
    dropped_reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "adapter_name": self.adapter_name,
            "workload_name": self.workload_name,
            "round_idx": self.round_idx,
            "metrics": dict(self.metrics),
            "run_succeeded": self.run_succeeded,
            "dropped_reason": self.dropped_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        return cls(
            adapter_name=d["adapter_name"],
            workload_name=d["workload_name"],
            round_idx=d["round_idx"],
            metrics=d["metrics"],
            run_succeeded=d["run_succeeded"],
            dropped_reason=d.get("dropped_reason"),
        )


@dataclass
class BenchResult:
    """Aggregate result from BenchmarkRunner.run() — N rounds."""
    adapter_name: str
    workload_name: str
    rounds: List[RunResult] = field(default_factory=list)

    def succeeded_rounds(self) -> List[RunResult]:
        return [r for r in self.rounds if r.run_succeeded]
