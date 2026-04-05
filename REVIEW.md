# GPT-5.4 Full Project Review

*Model: gpt-5.4 | Files reviewed: 17 | Date: 2026-04-05*

# Comprehensive Review of `rocm-trace-lite`

## Executive summary

This is a promising prototype with a real niche: **kernel-level ROCm tracing without roctracer/rocprofiler-sdk** is valuable, especially for bring-up, constrained environments, and dependency-sensitive deployments.

However, in its current form, I would classify it as **proof-of-concept, not production-ready**.

The core idea—**HSA runtime tool interception via `HSA_TOOLS_LIB` plus queue interception and profiling signals**—is directionally sound. But there are several correctness and robustness issues, and one major scalability problem:

- **The thread-per-dispatch completion model is the biggest architectural risk**
- **The current queue interception path likely mishandles completion semantics**
- **Timestamp domains are inconsistent between host-side roctx and GPU profiling timestamps**
- **There are memory leaks and lifecycle issues around queue callback userdata**
- **The tests validate schema and converter behavior, but not the actual C++ implementation**

Below is a detailed review.

---

# 1. Architecture & Design

## Overall assessment

The **HSA interception approach is fundamentally sound** for the stated scope:

- `OnLoad()` via `HSA_TOOLS_LIB`
- replacing selected HSA API table entries
- intercepting queue creation
- registering an AQL packet callback
- attaching profiling signals to kernel dispatch packets
- reading dispatch timestamps with `hsa_amd_profiling_get_dispatch_time`

That is a legitimate low-level design and is exactly the kind of mechanism you use when you want to avoid ROCm profiling stacks.

## What is good

### Sound choices
- **Using HSA tool API instead of LD_PRELOAD for HIP** is the right call.
- **Capturing at AQL dispatch level** gives you kernel timing independent of HIP/PyTorch frontend details.
- **SQLite RPD-compatible output** is a strong interoperability decision.
- **Built-in roctx shim** is a useful compatibility feature.

## Fundamental design concerns

## 1.1 Thread-per-dispatch is not viable at scale

**File:** `src/hsa_intercept.cpp`

Every kernel dispatch does:

```cpp
std::thread(signal_completion_handler, dd).detach();
```

This is the single biggest issue in the project.

### Why this is dangerous
For workloads with:
- many short kernels
- graph execution
- fused microkernels
- inference servers under concurrency

you can easily hit **tens of thousands to millions of dispatches**. A detached thread per dispatch means:

- huge thread creation/destruction overhead
- scheduler thrash
- memory pressure from thread stacks
- unbounded resource growth
- shutdown races with detached workers still touching global state

This is not just “suboptimal”; it is a likely failure mode.

### Recommendation
Use one of:
1. **`hsa_amd_signal_async_handler`** if supported/reliable in your target ROCm versions
2. **A bounded worker pool**
3. **A single or small number of poller threads** waiting on a queue of profiling signals

Your own `PLANNING.md` correctly identifies this. It should be **Priority 0**.

---

## 1.2 Completion signal semantics are likely wrong / unsafe

**File:** `src/hsa_intercept.cpp`

You replace the packet’s completion signal with your own profiling signal:

```cpp
modified.completion_signal = prof_signal;
```

Then after completion, you try to “forward” completion to the original signal:

```cpp
if (dd->original_signal.handle != 0) {
    hsa_signal_value_t orig_val = g_orig_core.hsa_signal_load_relaxed_fn(dd->original_signal);
    g_orig_core.hsa_signal_store_screlease_fn(dd->original_signal, orig_val - 1);
}
```

### Problem
This assumes the original completion signal semantics are equivalent to decrementing the signal value by 1 after your wait completes. That is not generally safe.

Potential issues:
- original signal may not have been initialized to 1
- original signal may be used by runtime/framework with different expected semantics
- store-based emulation may not preserve ordering/ownership expectations
- if multiple producers/consumers are involved, this can break synchronization

This is a **real correctness risk** and could corrupt application behavior.

### Recommendation
You need to verify the exact HSA completion signal contract for intercepted queue packets and whether:
- you can chain completion safely
- you should use a barrier/dependency packet instead
- you should preserve the original signal and add profiling another way
- AMD queue intercept API offers a supported profiling path without replacing completion semantics

As written, this is one of the most concerning parts of the implementation.

---

## 1.3 Timestamp domain mismatch

**Files:**  
- `src/rpd_lite.cpp`
- `src/roctx_shim.cpp`
- `src/hsa_intercept.cpp`

`roctx_shim.cpp` uses:

```cpp
tick() -> CLOCK_MONOTONIC
```

Kernel timestamps come from:

```cpp
hsa_amd_profiling_get_dispatch_time_fn(...)
```

These are not guaranteed to be in the same time domain.

### Consequence
In the same SQLite DB, you are mixing:
- host monotonic timestamps
- GPU profiling timestamps

If downstream tools assume a common timeline, the trace may be misleading or outright wrong.

### Recommendation
Either:
- explicitly document that host and GPU events are in different domains and should not be compared directly, or
- calibrate/translate GPU timestamps into host time using an HSA-supported clock correlation mechanism if available

Right now the README implies a unified trace experience that the implementation does not actually guarantee.

---

## 1.4 `queueId` is not a queue ID

**File:** `src/hsa_intercept.cpp`

You assign:

```cpp
dd->queue_id = user_que_idx;
```

This is almost certainly not a stable queue identifier. Your planning doc already acknowledges this.

### Impact
- output schema field is misleading
- converter has to heuristically collapse queue IDs
- multi-queue analysis is unreliable

This is acceptable for a prototype, but not for a profiler claiming RPD compatibility.

---

## 1.5 `gpuId` mapping is fragile

**File:** `src/hsa_intercept.cpp`

You infer `device_id` by storing an `int*` in queue registration userdata and matching `agent.handle` against `g_gpu_agents`.

This can work, but:
- it depends on `g_gpu_agents` ordering
- it leaks memory
- it is not robust across queue lifecycle
- it does not capture actual ROCm device identity beyond local vector index

For a lightweight tool, vector index may be acceptable, but it should be explicit.

---

## 1.6 Symbol capture via `hsa_executable_freeze` is reasonable but incomplete

**File:** `src/hsa_intercept.cpp`

Intercepting `hsa_executable_freeze` and enumerating symbols is a reasonable way to build a kernel object → name map.

Limitations:
- may miss symbols if code objects are loaded/frozen before tool setup
- may not handle dynamic unloading/reloading
- no handling of duplicate kernel object handles across executable lifetimes
- no cleanup of stale symbol mappings

Not fatal, but worth noting.

---

# 2. Code Quality

I’m focusing on **real bugs / correctness / safety issues**, not style.

---

## 2.1 Header comments are materially incorrect

**File:** `src/rpd_lite.h:1-13`

The header says:

- depends on `libamdhip64`, `libfmt`
- HIP interception via `LD_PRELOAD + dlsym(RTLD_NEXT)`
- copyright AMD 2026

None of that matches the actual project.

This is not just cosmetic; it misstates architecture and dependencies.

### Why it matters
- confuses maintainers and users
- can invalidate source-guard assumptions
- suggests this file was copied from another codebase and not reconciled

---

## 2.2 Memory leak: queue userdata never freed

**File:** `src/hsa_intercept.cpp`, in `my_hsa_queue_create`

```cpp
int* dev_idx = new int(0);
...
g_orig_ext.hsa_amd_queue_intercept_register_fn(*queue, queue_intercept_cb, dev_idx);
```

I see no corresponding delete.

### Impact
One leak per queue creation. In long-running services creating/destroying queues, this accumulates.

### Recommendation
Use a queue-associated object with explicit lifecycle, or maintain a map keyed by queue handle and clean it up on queue destroy interception.

---

## 2.3 Detached threads can outlive DB shutdown and global state

**Files:**  
- `src/hsa_intercept.cpp`
- `src/rpd_lite.cpp`

`OnUnload()` does:

```cpp
get_trace_db().flush();
get_trace_db().close();
```

But detached completion threads may still be running and may still call:

```cpp
get_trace_db().record_kernel(...)
```

### Consequences
- use-after-close behavior on SQLite handle guarded only by `if (!db_) return`
- lost records during shutdown
- races against finalized prepared statements
- races against global vectors/maps being torn down at process exit

The mutex helps only partially; it does not solve lifecycle coordination.

### Recommendation
You need:
- a global shutdown flag
- bounded worker threads
- join on shutdown
- stop accepting new work before DB close

---

## 2.4 Unsafe access to `g_gpu_agents` without lock

**File:** `src/hsa_intercept.cpp`, `signal_completion_handler`

```cpp
g_orig_ext.hsa_amd_profiling_get_dispatch_time_fn(
    g_gpu_agents[dd->device_id], dd->profiling_signal, &time);
```

`g_gpu_agents` is protected by `g_agent_mutex` during population, but read here without lock.

### Is this a bug?
In practice, `g_gpu_agents` is only populated during `OnLoad()`, before dispatches begin, so this is probably benign after initialization. But there is no formal immutability guarantee in code.

### Recommendation
Freeze it after init or make it const-after-load. At minimum, document that it is immutable after `OnLoad()`.

---

## 2.5 No status checking on critical HSA calls

**File:** `src/hsa_intercept.cpp`

Many calls ignore return status:
- `hsa_executable_symbol_get_info`
- `hsa_executable_iterate_symbols`
- `hsa_agent_get_info`
- `hsa_amd_profiling_set_profiler_enabled_fn`
- `hsa_amd_profiling_get_dispatch_time_fn`
- `hsa_signal_destroy_fn`

### Why this matters
If profiling is unsupported or partially supported on a queue/agent, you may write garbage timestamps or silently fail.

### Recommendation
Check and handle failures, especially:
- profiling enable
- dispatch time retrieval
- symbol info retrieval

---

## 2.6 `signal_completion_handler` ignores wait result

**File:** `src/hsa_intercept.cpp`

```cpp
hsa_signal_value_t val = g_orig_core.hsa_signal_wait_scacquire_fn(...);
```

`val` is unused, and no timeout/error handling exists.

### Problem
If the wait returns unexpectedly or profiling signal is invalid, you still proceed to query timestamps and record an op.

---

## 2.7 Potentially invalid packet type extraction

**File:** `src/hsa_intercept.cpp`

```cpp
uint8_t type = ((pkt->header >> HSA_PACKET_HEADER_TYPE) & 0xFF);
```

I would want to verify this against HSA packet header bit definitions. Usually packet type extraction uses mask/shift macros or exact field definitions. If `HSA_PACKET_HEADER_TYPE` is a shift constant, this may be okay; if not, it is wrong.

This is not definitely a bug from the snippet alone, but it is suspicious enough to audit.

---

## 2.8 `record_*` methods ignore `correlation_id`

**File:** `src/rpd_lite.cpp`

All of:
- `record_hip_api`
- `record_kernel`
- `record_copy`
- `record_roctx`

accept `correlation_id` but do not store it anywhere.

### Impact
- API suggests correlation support that does not exist
- impossible to link host and device events later
- misleading public contract

This is a real maintainability/correctness issue.

---

## 2.9 Prepared statement naming/reuse is misleading and error-prone

**File:** `src/rpd_lite.cpp`

You prepare:
- `stmt_api_` for string insert
- `stmt_kernel_` for API insert
- `stmt_copy_` for op insert

This works mechanically, but it is easy to misuse and already obscures intent.

Not a style nit: this increases bug probability and makes future extension risky.

---

## 2.10 `close()` may write duplicate metadata on repeated close paths

**File:** `src/rpd_lite.cpp`

You register `atexit()` in `lazy_init_db()`, and also call `close()` in `OnUnload()`.

`close()` guards with `if (!db_) return;`, so duplicate close is mostly okay. But metadata insertion happens every successful close before `db_ = nullptr`.

This is probably safe because only the first close executes, but the lifecycle is brittle and split across `atexit` and `OnUnload`.

---

## 2.11 SQLite API return values are mostly ignored

**File:** `src/rpd_lite.cpp`

Examples:
- `sqlite3_prepare_v2` return codes ignored
- `sqlite3_step` return codes ignored
- `sqlite3_exec` transaction calls ignored

### Consequence
DB corruption, readonly filesystem, disk full, schema mismatch, or lock issues will silently drop data.

For a profiler, silent data loss is bad.

---

## 2.12 `busy` view can exceed 100% for overlapping queues

**File:** `src/rpd_lite.cpp`, schema

```sql
ROUND(100.0 * SUM(end - start) / (MAX(end) - MIN(start)), 2) AS utilization_pct
```

If kernels overlap on multiple queues, this can exceed 100%.

### Impact
The test expects `<= 100`, but real traces may violate that.

This is a **semantic bug in the metric**, not just presentation.

---

## 2.13 `top` view includes roctx markers

**Files:**  
- `src/rpd_lite.cpp`
- `tests/test_roctx_shim.py`

The test itself acknowledges roctx markers may appear in `top`. That is undesirable for a “Top GPU Kernels” view.

### Recommendation
Filter `gpuId >= 0` and/or `opType='KernelExecution'`.

---

## 2.14 README test target is wrong

**File:** `README.md`

It says:

```bash
make test       # quick GPU smoke test (requires GPU)
```

But `Makefile` defines:

```make
test: test-cpu
```

This is a real documentation bug.

---

## 2.15 `rpd_lite.sh` likely fails after install

**File:** `tools/rpd_lite.sh`

It computes:

```bash
SCRIPT_DIR=...
LIB="${SCRIPT_DIR}/librpd_lite.so"
```

After installation, the script goes to `PREFIX/bin`, library to `PREFIX/lib`. `SCRIPT_DIR/librpd_lite.so` will not exist, so it falls back to `/usr/local/lib/librpd_lite.so`.

### Problem
If installed with non-default `PREFIX`, the script breaks unless the library also happens to be in `/usr/local/lib`.

### Recommendation
Use:
- `@PREFIX@/lib/librpd_lite.so` substitution at install time, or
- resolve relative to `../lib`, or
- rely on `ldconfig` and set `HSA_TOOLS_LIB=librpd_lite.so` only if discoverable

---

## 2.16 `ldconfig` in `make install` is not packaging-safe

**File:** `Makefile`

Running `ldconfig` unconditionally is inappropriate for:
- non-root installs
- staged installs
- packaging systems
- containers

This is a distribution bug.

---

## 2.17 `test-gpu` uses PyTorch CUDA naming on ROCm

**File:** `Makefile`

```make
python3 -c "import torch; x=torch.randn(...,device='cuda') ..."
```

This is actually okay for PyTorch ROCm, since it uses `cuda` API naming. No issue there.

---

# 3. Project Structure

## Overall
The intended structure is good:

- `src/`
- `tools/`
- `tests/`
- `examples/`

That is appropriate for this project size.

## Missing / inconsistent files

### 3.1 Missing GitHub workflow
`PLANNING.md` references:

- `.github/workflows/ci.yml`

but it is not present in the provided repo.

If this is the complete project, that is a missing file.

### 3.2 Missing LICENSE
`PLANNING.md` references `LICENSE`, but it is not present in the provided repo dump.

### 3.3 README examples use wrong paths
README says:

```bash
./rpd_lite.sh ...
python3 rpd2trace.py ...
```

But actual files are under `tools/`.

This is a structure/documentation mismatch.

### 3.4 Tests are mostly schema fixtures, not implementation tests
The `tests/` layout is fine, but the content does not test the actual C++ library. More on that below.

---

# 4. `PLANNING.md` Review

## Good
The roadmap is mostly realistic and shows awareness of current limitations.

Strong points:
- clear scope boundaries
- explicit non-goals
- recognition of queueId/gpuId issues
- recognition of thread-per-dispatch scalability problem
- phased packaging plan

## Concerns

### 4.1 Priority ordering is slightly off
The plan lists:
- queueId/gpuId fixes
- async copy
- thread pool / async handler

In reality, the order should be:

1. **Fix dispatch completion architecture**
2. **Validate correctness under real workloads**
3. **Fix timestamp domain / semantics**
4. **Fix queueId/gpuId fidelity**
5. Then async copy / packaging

### 4.2 Overhead target is optimistic
`< 5%` and later `< 2%` are not realistic with the current architecture. Even with a better completion model, HSA interception plus SQLite writes can still be noticeable on dispatch-heavy workloads.

### 4.3 “Skip build test in free CI” is understandable but risky
This means the main implementation can regress without any CI signal. For a C++ systems project, that is dangerous.

### 4.4 Missing compatibility matrix
The plan should explicitly track:
- ROCm versions supported
- kernel/driver combinations
- MI2xx/MI3xx coverage
- framework coverage (PyTorch, HIP samples, Triton, vLLM, etc.)

### 4.5 Missing crash-consistency plan
Since SQLite is used with:
- `journal_mode=WAL`
- `synchronous=OFF`

you should explicitly discuss trace durability and partial-trace recovery.

---

# 5. Testing

## Current non-GPU tests are not sufficient

They are useful, but they mostly test:
- schema shape
- synthetic DB contents
- converter behavior
- source text guards

They do **not** validate:
- that the C++ code builds
- that the C++ code writes the schema it claims
- that roctx shim symbols work
- that HSA interception works
- that queue interception preserves correctness
- that shutdown is safe

## Critical gaps

### 5.1 No build test
There is no CI build of the shared library.

This is the most obvious gap.

### 5.2 No native unit tests for `TraceDB`
You could test `TraceDB` without GPU by compiling a small native test binary and verifying:
- schema creation
- inserts
- flush/close behavior
- concurrent writes
- duplicate close
- output file path handling

### 5.3 No native tests for roctx shim
Current `test_roctx_shim.py` does not test `roctx_shim.cpp`; it tests a Python simulation of expected DB rows.

That is not a real implementation test.

### 5.4 No stress tests
You need at least synthetic stress tests for:
- 100K+ record inserts
- concurrent recorders
- shutdown during active writes

### 5.5 No correctness tests for queue interception
The GPU smoke test is too weak. You need:
- correctness of application output under tracing
- no hangs
- no deadlocks
- no missing completion
- multiple queues/streams
- multi-threaded host dispatch

## CI strategy assessment

### Good
- non-GPU tests on every push are cheap and useful

### Not enough
For a systems tool, I would want:
- CPU-only native build job
- optional ROCm-header container build job
- self-hosted GPU integration job on PRs or nightly

Even if free-tier CI cannot install full ROCm, you can still:
- compile against a container image
- or at least run static analysis / syntax checks

---

# 6. Packaging & Distribution

## Build system
For now, **Make is acceptable**. The project is small.

But for wider adoption, you will likely need:
- CMake or Meson
- install-time path configuration
- versioned SONAME
- pkg-config metadata
- proper dependency discovery

## Specific issues

### 6.1 Current install path handling is brittle
As noted, `tools/rpd_lite.sh` is not relocatable.

### 6.2 No SONAME/versioning in shared library
`librpd_lite.so` is built without version metadata. Fine for internal use, weak for packaging.

### 6.3 No `pkg-config`, no package metadata
Needed if others will integrate or redistribute.

### 6.4 Python packaging plan is premature
A pip wheel for a ROCm/HSA-intercepting `.so` is possible, but manylinux + ROCm ABI/versioning is nontrivial. I would defer that until:
- architecture stabilizes
- compatibility matrix exists
- GPU CI exists

### 6.5 `ldd` dependency check is useful but incomplete
Good sanity check, but:
- `ldd` is runtime-loader dependent
- transitive runtime loading via `dlopen` won’t show
- should not be the only dependency guard

---

# 7. Security & Robustness

## Thread safety
### Good
- DB writes are mutex-protected
- correlation ID uses atomic
- roctx range state is thread-local

### Problems
- detached worker lifecycle is unsafe
- queue userdata leak
- no coordinated shutdown
- global state accessed during process teardown

## Memory leaks
### Confirmed
- `new int(0)` in `my_hsa_queue_create` leaked
- possible stale symbol map growth over long process lifetime

## Signal handling
No Unix signal handling is implemented, which is okay, but then don’t imply crash-safe post-mortem behavior. With `synchronous=OFF`, abrupt termination can lose recent trace data.

## Crash recovery
Weak:
- WAL + `synchronous=OFF` prioritizes speed over durability
- no periodic checkpointing
- no recovery marker / incomplete-trace metadata

## Input robustness
### `rpd2trace.py`
Mostly okay for trusted local files, but:
- loads all events into memory
- no streaming output
- no schema version check
- no malformed DB handling beyond basic exceptions

For large traces, memory usage may become significant.

---

# 8. Competitive Analysis

## vs `rocprofiler` / `rocprofiler-sdk`
### Advantages
- much smaller dependency surface
- simpler deployment
- potentially useful when official profiler stack is unavailable or broken
- focused kernel timing path

### Disadvantages
- much less feature complete
- no counters
- no PC sampling
- no API tracing parity
- likely weaker correctness guarantees
- weaker maintenance story across ROCm versions

## vs NVIDIA Nsight Systems
Nsight Systems provides:
- mature timeline correlation
- CPU/GPU/API integration
- robust UI
- lower-risk production workflows

`rocm-trace-lite` cannot compete on breadth or polish.

## vs Intel VTune
VTune similarly offers:
- richer host/device correlation
- mature analysis workflows
- stronger tooling ecosystem

## Unique value proposition
The project’s real differentiator is:

> **“I need a minimal ROCm kernel profiler that works with only HSA runtime and SQLite, with no roctracer/rocprofiler-sdk dependency.”**

That is a legitimate and useful niche.

This is strongest for:
- bring-up
- constrained CI
- debugging on partially provisioned systems
- embedding lightweight tracing in custom environments

That should be the primary positioning—not “replacement for RPD,” but “minimal fallback backend / lightweight kernel tracer.”

---

# 9. Top 5 Risks

## 1. Unbounded thread creation causes hangs, overhead, or process instability
**Most likely operational failure.**

## 2. Completion signal forwarding breaks application synchronization
**Most serious correctness risk.**

## 3. Mixed timestamp domains produce misleading traces
Users may draw wrong conclusions from host/GPU event alignment.

## 4. ROCm/HSA API behavior changes across versions
This design depends on lower-level interception details that may be less stable than official profiler APIs.

## 5. Lack of real implementation tests allows regressions to ship unnoticed
Especially because current tests mostly validate synthetic data, not the actual tracer.

---

# 10. Top 5 Recommended Next Actions

## 1. Replace thread-per-dispatch with a bounded completion architecture
Highest priority.

Preferred order:
- evaluate `hsa_amd_signal_async_handler`
- otherwise implement a fixed worker pool or poller thread model

Do this before adding features.

## 2. Audit and fix completion signal semantics
Prove that replacing and forwarding `completion_signal` is correct, or redesign the mechanism.

This needs validation against:
- HSA spec
- ROCm runtime behavior
- PyTorch/HIP workloads under concurrency

## 3. Add native non-GPU tests for `TraceDB` and roctx shim, plus a build job
At minimum:
- compile the library in CI
- test DB open/write/flush/close
- test roctx exported symbols from a tiny C/C++ harness
- test concurrent writes

## 4. Fix lifecycle/resource management
Specifically:
- eliminate leaked queue userdata
- add shutdown coordination
- ensure no worker can write after DB close
- check all critical HSA/SQLite return codes

## 5. Clean up schema semantics and documentation
Fix:
- `top` view filtering
- `busy` utilization semantics
- queueId naming/meaning
- README path/test inaccuracies
- header comments in `src/rpd_lite.h`

---

# File-specific findings

## `Makefile`
- Good: simple and readable
- Bug: `install` runs `ldconfig` unconditionally
- Bug: README says `make test` is GPU smoke test, but `test` is CPU-only
- Missing: no SONAME/versioning
- Missing: no build-time dependency detection for ROCm headers

## `src/hsa_intercept.cpp`
- Major scalability issue: detached thread per dispatch
- Likely correctness issue: manual completion signal forwarding
- Leak: `new int(0)` userdata never freed
- Weak error handling on HSA calls
- queueId semantics incorrect
- gpuId mapping simplistic

## `src/rpd_lite.cpp`
- SQLite return codes mostly ignored
- `busy` view can exceed 100%
- `top` view includes non-kernel markers
- correlation IDs unused
- prepared statement naming misleading
- lifecycle split between `atexit` and `OnUnload`

## `src/rpd_lite.h`
- Header comment is stale/incorrect and contradicts implementation

## `src/roctx_shim.cpp`
- Thread-local design is okay
- Real implementation not actually tested by current test suite
- `roctxRangeStop` only works on same thread as start; if that matches roctx semantics, okay, otherwise document it

## `tools/rpd2trace.py`
- Functional and straightforward
- Loads all events into memory; may not scale to very large traces
- Queue collapse heuristic is pragmatic but compensates for upstream schema misuse
- Empty trace behavior is weakly specified; should return explicit nonzero or documented zero

## `tools/rpd_lite.sh`
- Not relocatable after non-default install prefix
- Reasonable UX otherwise

## `tests/*`
- Good synthetic schema coverage
- Not sufficient as implementation validation
- `test_roctx_shim.py` is not actually testing the shim
- `test_schema.py` duplicates schema text from implementation, which can mask divergence if both are updated manually but incorrectly

---

# Final verdict

## Is the project worth continuing?
**Yes.** The concept is valid and useful.

## Is the current implementation sound enough for production use?
**No.** Not yet.

## Main blocker
The combination of:
- **thread-per-dispatch**
- **questionable completion signal forwarding**
- **weak lifecycle management**

means I would not trust this in a high-throughput or correctness-sensitive environment yet.

## Best framing
This should currently be presented as:

> **Experimental lightweight ROCm kernel tracer using HSA interception**

not yet as a robust drop-in profiler.

If you want, I can next provide:
1. a **line-by-line annotated review**, or  
2. a **proposed redesign for the dispatch completion path**.