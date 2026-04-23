"""Conftest for profiler_perf_bench tests.

Ensures profiler_perf_bench and rocm_trace_lite are importable regardless
of install method. Both packages live in the same repo root.
"""

import sys
import os
from pathlib import Path

# Repo root = two levels up from this file:
#   profiler_perf_bench/tests/conftest.py → profiler_perf_bench/ → repo_root/
_REPO_ROOT = Path(__file__).parent.parent.parent

# Worktree root (agent worktree for this session)
_WORKTREE_ROOT = Path(__file__).parent.parent.parent.parent / ".claude" / "worktrees" / "agent-a4795220"

for _p in [str(_REPO_ROOT), str(_WORKTREE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
