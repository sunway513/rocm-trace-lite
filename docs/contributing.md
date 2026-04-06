# Contributing

## Development setup

```bash
git clone https://github.com/sunway513/rocm-trace-lite.git
cd rocm-trace-lite
make -j
pip install -e .
```

## Running tests

```bash
# CPU-only tests (no GPU required, 144 tests)
cd tests && python3 -m pytest -v

# GPU integration tests (requires ROCm GPU)
make test-gpu
```

## Code style

- **C++**: C++17, `-O3 -g`, no exceptions in hot paths
- **Python**: ruff-clean, pytest for tests
- **Commits**: imperative mood, explain "why" not "what"
- **GitHub**: all issues, PRs, and commit messages in English

## Before submitting a PR

1. Run `ruff check` on Python files
2. Run `python3 -m pytest -v` in tests/
3. Build and verify clean dependency chain (`make`)
4. Run GPU tests if you modified C++ code

## Architecture guidelines

- Keep the .so dependency chain minimal (HSA + SQLite only)
- No `#include` of roctracer, rocprofiler-sdk, or libroctx64 headers
- Source guard tests enforce this automatically
