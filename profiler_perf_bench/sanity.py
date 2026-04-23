"""Sanity guards — 4 fixed pre-condition checks after each run_once().

Per spec §3.4, exactly 4 rules, no more. If a 5th rule is needed,
file a TODO(sanity-rev) comment and stop at 4.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .workloads.base import Level


@dataclass
class SanityResult:
    """Result of running all 4 sanity checks."""
    run_succeeded: bool
    dropped_reason: Optional[str]


def check_sanity(
    exit_code: int,
    adapter_name: str,
    workload_level: Level,
    artifact_dir: Path,
    artifact_glob: str,
    metrics: dict,
    l3_successful_requests: Optional[int],
) -> SanityResult:
    """Run all 4 sanity checks in order. Returns on first failure.

    Rules (spec §3.4):
      1. subprocess exit == 0
      2. If adapter != "none": artifact glob matches ≥1 file
      3. If artifact present: file size > 100 bytes
      4. If workload.level == L3: successful_requests > 0
    """

    # Rule 1: exit code
    if exit_code != 0:
        return SanityResult(run_succeeded=False, dropped_reason="crashed")

    # Rules 2 + 3: trace file check (only for non-none adapters)
    if adapter_name != "none" and artifact_glob:
        import glob as _glob
        matches = list(_glob.glob(str(artifact_dir / "**" / artifact_glob), recursive=True))
        if not matches:
            # Also try non-recursive glob
            matches = list(_glob.glob(str(artifact_dir / artifact_glob)))
        if not matches:
            return SanityResult(run_succeeded=False, dropped_reason="no_trace_produced")

        # Rule 3: check first matched file size
        total_bytes = sum(Path(m).stat().st_size for m in matches if Path(m).is_file())
        if total_bytes <= 100:
            return SanityResult(run_succeeded=False, dropped_reason="corrupt_trace")

    # Rule 4: L3 server must have served at least one request
    if workload_level == Level.L3:
        if l3_successful_requests is None or l3_successful_requests <= 0:
            return SanityResult(run_succeeded=False, dropped_reason="server_never_served")

    return SanityResult(run_succeeded=True, dropped_reason=None)
