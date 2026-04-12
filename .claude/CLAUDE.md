# rocm-trace-lite — Claude Code Instructions

## Benchmark Testing
- Use `/gpu-bench` skill for ALL remote GPU benchmark work — no exceptions
- Always verify installed librtl.so matches source after wheel install (md5sum)
- Always capture RTL diagnostic counters (intercept/inject/skip) after profiled runs

## Build
- `make` outputs librtl.so to repo root; setup.py packages from there
- Always `make clean && make` before `python3 setup.py bdist_wheel`
- Run `python3 -m pytest tests/ -v --tb=short` before any commit

## Known Issues
- RTL lite mode: ~0% overhead on MoE models with CUDAGraph, ~5-6% on enforce-eager
- RTL hip mode: GPU_CLR_PROFILE_OUTPUT + CUDAGraph replay = crash (issue #79)
- roctracer: ROCP_OUTPUT_DIR must be set or stdout floods cause OOM

## Code Style
- All GitHub issues/PRs/commits in English
- Run `ruff check` before push
