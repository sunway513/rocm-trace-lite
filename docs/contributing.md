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
# CPU-only tests (no GPU required, 314+ tests)
cd tests && python3 -m pytest -v

# GPU integration tests (requires ROCm GPU)
make test-gpu

# HSA table interception conformance (no GPU, no real HSA runtime)
make test-mock-hsa
```

## Code style

- **C++**: C++17, `-O3 -g`, no exceptions in hot paths
- **Python**: ruff-clean, pytest for tests

### Testing HSA interception without a GPU

`tests/mock_hsa_table.cpp` loads the built `librtl.so` through `OnLoad()` with a
synthetic `HsaApiTable` whose entries are recording stubs, then drives the
installed hooks directly. It covers the parts of interception that are easy to
get wrong and hard to see on real hardware: API-table ABI guards when RTL is
built against newer ROCm headers than the runtime it loads into, and whether
every `hsa_amd_queue_create` descriptor field reaches the queue that is actually
created.

Each phase runs in its own child process, because `OnLoad()` may only be called
once per process. Checks that need an API the installed headers do not have
report `SKIP` rather than failing, so the same test runs on every supported ROCm
version.
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
