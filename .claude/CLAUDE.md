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
- RTL lite mode: ~0% overhead on MoE models with CUDAGraph, ~3-5% on enforce-eager
- RTL hip mode: GPU_CLR_PROFILE_OUTPUT + CUDAGraph replay = crash (issue #79)
- Container restart during `pkill -f openai_server` kills RTL trace process too — use targeted PID kill, not pkill pattern match

## ATOM Server Entry Points
- ATOM image `rocm/atom:rocm7.2.0-*`: `python3 /app/ATOM/atom/entrypoints/openai_server.py` (NOT `python3 -m atom.serve`)
- Benchmark: `python3 -m atom.benchmarks.benchmark_serving` (works as module)
- Benchmark needs `--tokenizer /path/to/model` for local models (HuggingFace name resolution fails offline)

## Release Process
- Bump version in both `pyproject.toml` AND `rocm_trace_lite/__init__.py`
- Update `docs/changelog.md`
- Commit, tag `v0.X.Y`, push tag — triggers `.github/workflows/release.yml`
- Release workflow builds wheel in ROCm 7.2 container, creates GitHub Release

## Code Style
- All GitHub issues/PRs/commits in English
- Run `ruff check` before push
