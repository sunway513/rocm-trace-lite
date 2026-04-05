# rocm-trace-lite — Project Planning

## 1. Project Scope

### What it IS
- A **single .so library** that captures GPU kernel execution traces on ROCm
- Self-contained: depends only on HSA runtime + SQLite (no roctracer, no rocprofiler-sdk)
- Drop-in profiler: `HSA_TOOLS_LIB=librpd_lite.so python my_workload.py`
- Outputs standard RPD SQLite format, compatible with existing RPD ecosystem tools
- Includes Perfetto/Chrome trace converter

### What it is NOT
- Not a replacement for rocprofiler (no HW counters, no PC sampling)
- Not a HIP API tracer (that requires roctracer; we capture at HSA level only)
- Not a real-time monitoring tool (post-mortem trace analysis)

### Target users
- AI framework teams (ATOM, vLLM, SGLang) who need clean kernel profiling without roctracer overhead/dependency
- Bring-up engineers profiling on new hardware where rocprofiler may not be ready
- CI systems that need lightweight regression testing of kernel performance

---

## 2. Repository Structure

```
rocm-trace-lite/
├── README.md                 # Quick start, usage, examples
├── PLANNING.md               # This file — project decisions and rationale
├── LICENSE                   # MIT
├── Makefile                  # Build system (single .so target)
│
├── src/                      # C++ source
│   ├── rpd_lite.h            # Public API: TraceDB, tick(), correlation IDs
│   ├── rpd_lite.cpp          # SQLite trace DB implementation
│   ├── hsa_intercept.cpp     # HSA_TOOLS_LIB OnLoad, queue intercept, symbol resolution
│   ├── roctx_shim.cpp        # Built-in roctx API implementation
│   └── hip_intercept.cpp     # Placeholder (documents why HIP interception is not used)
│
├── tools/                    # Python utilities
│   ├── rpd2trace.py          # RPD SQLite → Perfetto/Chrome JSON converter
│   └── rpd_lite.sh           # Launcher script
│
├── tests/                    # Test suite
│   ├── conftest.py           # Shared fixtures, synthetic data generators
│   ├── test_source_guard.py  # T1: dependency guard (no roctracer refs)
│   ├── test_schema.py        # T2: SQLite schema and data integrity
│   ├── test_roctx_shim.py    # T3: roctx record format validation
│   ├── test_rpd2trace.py     # T5: converter correctness
│   └── test_gpu.py           # T4: GPU integration tests (requires ROCm GPU)
│
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions: non-GPU tests on every push
│
└── examples/                 # Usage examples
    ├── trace_matmul.py       # Minimal: trace a single matmul
    └── trace_model.py        # Model inference tracing with roctx markers
```

### Decision: flat src/ vs current flat layout
**Move source files into `src/`** — separates build artifacts from source, cleaner for packaging. Makefile `VPATH` handles it.

### Decision: tools/ directory
**rpd2trace.py and rpd_lite.sh go in `tools/`** — they're not part of the .so build, cleaner separation.

---

## 3. Build System

### Current: GNU Make (keep it)
- Single .so output, no complex dependency graph
- CMake is overkill for 4 source files
- Makefile targets:

```makefile
all:            # build librpd_lite.so
install:        # copy .so to PREFIX/lib, tools to PREFIX/bin
clean:          # remove build artifacts
test-cpu:       # run non-GPU pytest suite
test-gpu:       # run GPU integration tests (requires ROCm GPU)
test:           # alias for test-cpu
```

### Dependencies at build time
- ROCm headers (`hsa/hsa.h`, `hsa/hsa_api_trace.h`) — available in `rocm-dev` package
- SQLite3 dev headers (`libsqlite3-dev`)
- g++ with C++17

### Dependencies at runtime
- `libhsa-runtime64.so` (part of ROCm runtime, always present)
- `libsqlite3.so`
- NO: libroctracer64, librocprofiler-sdk, libroctx64, libamdhip64

### Build verification
Makefile `all` target runs `ldd` check to verify no forbidden deps linked.

---

## 4. Testing Strategy

### Layer 1 — Non-GPU (GitHub Actions, every push)

| Suite | Tests | What it guards |
|-------|-------|----------------|
| test_source_guard.py | 6 | No roctracer/rocprofiler includes, symbols, linker flags |
| test_schema.py | 18 | SQLite schema creation, kernel records, string dedup, views |
| test_roctx_shim.py | 6 | roctx record format, opType, gpuId, nesting |
| test_rpd2trace.py | 11 | JSON conversion, event fields, edge cases, Perfetto format |
| **Total** | **41** | |

### Layer 2 — GPU Integration (self-hosted or manual, on-demand)

| Test | What it validates |
|------|-------------------|
| OnLoad/OnUnload lifecycle | Library loads, creates trace file, finalizes |
| Kernel capture | GEMM kernel appears in trace with "Cijk" in name |
| Timing sanity | All ops have end > start, duration < 10s |
| No computation corruption | Output matches baseline without profiler |
| Multi-GPU | gpuId correctly distributed across devices |

### Layer 3 — Regression (nightly, requires model weights)

| Test | What it validates |
|------|-------------------|
| Model inference trace | >1K kernel ops from real model forward pass |
| Baseline comparison | Same workload: rpd_lite kernel count within 1% of original RPD |
| Performance overhead | Wall-clock time with tracer < 5% slower than without |

### Decision: pytest for everything
Python tests for all layers. GPU tests use subprocess to launch traced workloads, then inspect the SQLite output. No C++ test framework needed.

---

## 5. CI / CD

### GitHub Actions (free tier)

```yaml
on push to main:
  - source-guard     (2s, ubuntu-latest)
  - python-tests     (10s, ubuntu-latest, pytest)
  - lint             (5s, ubuntu-latest, ruff + bash -n)
```

### Future: self-hosted GPU runner
- Add `test-gpu` job on self-hosted runner with MI300X
- Trigger on PR to main, not every push
- Cache ROCm Docker image to speed up

### Decision: no Docker-based build in CI yet
Building the .so requires ROCm headers. Options:
1. ~~Install ROCm on runner~~ — too slow (3GB+ download)
2. ~~Use `rocm/dev-ubuntu-24.04` Docker~~ — works but heavy for free tier
3. **Skip build test in free CI, validate at runtime on GPU** — pragmatic for now

Build verification happens during GPU integration tests on self-hosted.

---

## 6. Packaging & Distribution

### Phase 1 (now): source-only
```bash
git clone https://github.com/sunway513/rocm-trace-lite
cd rocm-trace-lite && make
```

### Phase 2: pip-installable Python wrapper
```
pip install rocm-trace-lite
```
- Python package installs `rpd2trace.py` and `rpd_lite.sh` as CLI tools
- Pre-built .so wheels for ROCm 6.x and 7.x (manylinux)
- `setup.py` / `pyproject.toml` with C extension build

### Phase 3: integration into ROCm packaging
- Contribute back to ROCm/rocmProfileData as an alternative backend
- Or package as standalone ROCm tool (`apt install rocm-trace-lite`)

### Decision: defer packaging complexity
Phase 1 only for now. `make && make install` is sufficient for internal use.

---

## 7. Known Limitations & Issues

| Issue | Status | Priority |
|-------|--------|----------|
| queueId is per-dispatch write index, not actual queue ID | Open (#1 on fork) | High |
| gpuId always 0 (single GPU per process) | Open (#3 on fork) | High |
| No async memory copy tracking | Open (#2 on fork) | Medium |
| No HIP API tracing | By design | — |
| No HW performance counters | Out of scope | — |
| detached threads for signal wait (one per dispatch) | Scalability risk | Medium |

### Thread-per-dispatch concern
Current design spawns `std::thread` per kernel dispatch to wait on the profiling signal. For workloads with 100K+ dispatches, this creates massive thread churn. Fix:
- **Option A**: Thread pool (ctpl or fixed-size) — bounded resource usage
- **Option B**: Async signal wait with `hsa_signal_wait_scacquire` in a single poller thread
- **Option C**: Use HSA AMD signal handler callback (`hsa_amd_signal_async_handler`)

**Decision**: Implement Option C (async handler) as priority fix after initial validation.

---

## 8. Versioning

Semantic versioning: `MAJOR.MINOR.PATCH`

- `0.1.0` — Initial release: HSA kernel tracing, SQLite output, rpd2trace converter
- `0.2.0` — Fix queueId/gpuId, add async copy, thread pool
- `0.3.0` — pip packaging, CI with GPU tests
- `1.0.0` — Production-ready, multi-GPU validated, performance overhead < 2%

---

## 9. Code Style & Conventions

- C++17, `-O3 -g` (always with debug symbols for post-mortem)
- No exceptions in hot path (signal handlers, intercept callbacks)
- All external-facing C functions: `extern "C"` with C linkage
- Python: ruff-clean, pytest for tests
- Commit messages: imperative mood, explain "why" not "what"
- GitHub issues: English (per project convention)

---

## 10. Relationship to ROCm/rocmProfileData

rocm-trace-lite is a **standalone alternative**, not a fork:
- Does NOT depend on any RPD code (written from scratch)
- Outputs RPD-compatible SQLite schema (interoperable)
- Can coexist with original RPD on the same system
- Goal: upstream as an optional "lite" backend in RPD if the community finds it useful

The sunway513/rocmProfileData fork (rpd_lite branch) is archived — all future development happens in this repo.
