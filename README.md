# rocm-trace-lite

Self-contained GPU kernel profiler for ROCm. **Zero roctracer/rocprofiler-sdk dependency.**

## What it does

A streamlined, lightweight GPU kernel profiler. Captures dispatch timestamps using only HSA runtime interception (`HSA_TOOLS_LIB`), writing to a standard SQLite `.db` file. No dependency on HIP, roctracer, or rocprofiler-sdk.

### Comparison with other ROCm profiling tools

| Feature | rocm-trace-lite | [RPD](https://github.com/ROCm/rocmProfileData) | [rocprofiler-sdk](https://rocm.docs.amd.com/projects/rocprofiler-sdk) | [roctracer](https://rocm.docs.amd.com/projects/roctracer) | [Triton Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton) |
|---------|----------------|-----|-----------------|-----------|---------------|
| **Dependencies** | libhsa-runtime64 + libsqlite3 | + libroctracer64 | Full ROCm 6.0+ stack | libroctracer64 | libroctracer64 (AMD) |
| **GPU kernel timing** | HSA signal injection | roctracer activity | Buffered/callback tracing | Activity records | roctracer / CUPTI |
| **HIP API tracing** | Yes (`RTL_MODE=hip`) | Yes | Yes | Yes | — |
| **HSA API tracing** | — | — | Yes | Yes | — |
| **roctx markers** | Built-in shim | Via roctracer | Native | Yes (libroctx64) | Indirect |
| **HW counters** | — | — | Yes (AQLprofile) | — | NVIDIA only |
| **Output format** | SQLite (.db) + rocprofv3 JSON | SQLite (.rpd) | CSV / JSON / Perfetto / OTF2 | Raw callbacks | JSON / Chrome Trace |
| **Perfetto visualization** | `rtl convert` | rpd2tracing.py | Native PFTrace | — | Built-in |
| **TraceLens compatible** | Yes (`--format rocprofv3`) | No | Yes (native) | No | No |
| **Zero profiler dependency** | Yes | No | No | No | No |
| **Status** | Active | Active | Active (recommended) | Legacy (EoS 2026 Q2) | Active |

## Installation

### From wheel (recommended)

Download the latest `.whl` from [GitHub Releases](https://github.com/sunway513/rocm-trace-lite/releases):

```bash
# Install the latest release
pip install rocm-trace-lite --find-links https://github.com/sunway513/rocm-trace-lite/releases/latest

# Or download and install manually
wget https://github.com/sunway513/rocm-trace-lite/releases/latest/download/rocm_trace_lite-<version>-py3-none-linux_x86_64.whl
pip install rocm_trace_lite-*.whl
```

After installation, the `rtl` CLI command is available. One command does everything — trace, summary, and Perfetto export:

```bash
rtl trace python3 my_model.py
```

### From source

```bash
# Build (requires ROCm headers)
make -j

# Install system-wide
make install    # copies librtl.so to /usr/local/lib, scripts to /usr/local/bin
```

Requirements:
- ROCm (for HSA headers: `hsa/hsa.h`, `hsa/hsa_api_trace.h`)
- SQLite3 development headers (`apt install libsqlite3-dev`)
- g++ with C++17

## Quick start

```bash
rtl trace python3 my_model.py                        # lite mode (default)
rtl trace --mode standard python3 my_model.py        # standard mode (~2-4% overhead)
rtl trace --mode hip python3 my_model.py             # hip mode (HIP API + GPU timing)
```

### Profiling modes

| Mode | GPU timing | HIP API | Graph replay | Overhead | Use case |
|------|-----------|---------|-------------|----------|----------|
| **lite** | Yes (partial) | No | Skipped | ~0% | Production / always-on **(default)** |
| **standard** | Yes | No | Skipped | ~2-4% | General profiling |
| **hip** | Yes | Yes | Skipped | <1% | CPU+GPU correlation |
| **full** | Yes (all) | No | Profiled | ~2-5% | Deep analysis (requires ROCm 7.13+) |

Set via CLI (`--mode`) or env var (`RTL_MODE=lite`).

**lite** skips packets that already have a completion signal (e.g., NCCL kernels, barriers), resulting in near-zero overhead and safety on ROCm <= 7.2. This is the default when `--mode` is not specified. **standard** mode profiles all count==1 dispatches including those with signals. **full** profiles everything including CUDAGraph replay batches, but requires ROCm 7.13+ to avoid a [known ROCR heap overflow](https://github.com/sunway513/rocm-trace-lite/issues/67).

Sample output:

```text
rtl: loading (HSA runtime v3)
rtl: lazy init, writing to trace_12345.db
rtl: found 1 GPU agent(s)
rtl: signal pool initialized (64 signals)
rtl: completion worker started

Trace: trace.db (728 GPU ops)

Kernel                                                       Calls  Total(us)  Avg(us)      %
================================================================================================
Cijk_Ailk_Bljk_HHS_BH_MT128x128x128                           240    28252.9    117.7   21.8
ncclDevKernel_Generic                                          160    29747.8    185.9   23.0
__amd_rocclr_fillBufferAligned.kd                             7900    27929.8      3.5   21.6

GPU Utilization:
  GPU 0: 0.13% (2630 ops, 17.2ms busy)
  GPU 1: 0.11% (2430 ops, 15.0ms busy)

Output files:
  trace.db
  trace_summary.txt
  trace.json.gz (1.2 MB → open in https://ui.perfetto.dev)
```

## TraceLens integration

RTL traces can be analyzed with [TraceLens](https://github.com/AMD-AGI/TraceLens) for automated performance reports — kernel breakdown, GPU timeline, roofline metrics, and more.

```bash
# 1. Collect trace
rtl trace -o trace.db python3 my_model.py

# 2. Convert to rocprofv3 format
rtl convert trace.db --format rocprofv3 -o trace_results.json

# 3. Generate TraceLens report
pip install git+https://github.com/AMD-AGI/TraceLens.git
TraceLens_generate_perf_report_rocprof --profile_json_path trace_results.json --kernel_details
```

This produces an Excel workbook with GPU timeline breakdown, kernel summary by category (GEMM, Elementwise, Reduction, etc.), and per-dispatch details with grid/block dimensions. Validated on GPT-OSS 120B TP=8 (162K dispatches, 92 unique kernels). See [issue #100](https://github.com/sunway513/rocm-trace-lite/issues/100) for sample output.

## How it works

1. **HSA_TOOLS_LIB OnLoad** — ROCm HSA runtime calls `OnLoad()` when the library is loaded, giving us the HSA API table
2. **Queue intercept** — We replace `hsa_queue_create` to create interceptible queues via `hsa_amd_queue_intercept_create`, then register a callback on every AQL packet
3. **Kernel profiling** — For each kernel dispatch packet, we insert a profiling signal, wait for completion, then read GPU timestamps via `hsa_amd_profiling_get_dispatch_time`
4. **Symbol resolution** — We intercept `hsa_executable_freeze` to enumerate kernel symbols from code objects
5. **roctx shim** — Provides `roctxRangePushA`/`roctxRangePop`/`roctxMarkA`/`roctxRangeStartA`/`roctxRangeStop` symbols so applications using roctx markers get captured without linking libroctx64

## Output format

Standard SQLite `.db` database. Query with any SQLite tool. Key tables:

- `rocpd_op` — GPU kernel dispatches with start/end timestamps, gpuId, queueId
- `rocpd_string` — Deduplicated string table (kernel names, op types)
- `rocpd_metadata` — Trace metadata (duration, host info)

Built-in views:

```sql
-- Top kernels by total GPU time
SELECT * FROM top LIMIT 10;

-- GPU utilization per device
SELECT * FROM busy;
```

## Tests

314 tests covering unit, E2E, multi-GPU, stress, and release validation.

```bash
# CPU-only tests (no GPU required)
make test-cpu

# GPU tests (requires ROCm GPU)
python3 -m pytest tests/ -v --timeout=180

# CI: CPU on every push, GPU on MI355X runners
```

## Embedding

rocm-trace-lite can be embedded into another profiler or tracer by redirecting events through callback hooks instead of writing to the built-in SQLite database. This is how [rocmProfileData (RPD)](https://github.com/ROCm/rocmProfileData) integrates rocm-trace-lite as its `RtlDataSource`.

### Callback API

Register callbacks **before** HSA `OnLoad` fires. In practice this means setting them during your library's initialization, before the application makes its first HIP/HSA call:

```cpp
#include "trace_db.h"

// Kernel dispatch completion (GPU-side, called from the completion worker thread)
void my_kernel_handler(const trace_db::KernelEventRecord& event, void* user_data) {
    // event.name         — demangled kernel name
    // event.device_id    — GPU index
    // event.queue_id     — HSA queue handle
    // event.start_ns     — GPU start timestamp (ns)
    // event.end_ns       — GPU end timestamp (ns)
    // event.correlation_id
    // event.wg_x/y/z     — workgroup dimensions
    // event.grid_x/y/z   — grid dimensions
}

// HIP API call (CPU-side, called from the app thread, requires RTL_MODE=hip)
void my_api_handler(const trace_db::ApiEventRecord& event, void* user_data) {
    // event.name         — e.g. "hipModuleLaunchKernel"
    // event.args         — formatted argument string
    // event.start_ns     — host start timestamp (ns, CLOCK_MONOTONIC)
    // event.end_ns       — host end timestamp (ns)
    // event.correlation_id
    // event.pid, event.tid
}

// 1. Register callbacks
trace_db::set_kernel_event_callback(my_kernel_handler, nullptr);
trace_db::set_api_event_callback(my_api_handler, nullptr);

// 2. ... application runs, callbacks fire ...

// 3. At shutdown, drain pending events before finalizing your storage
trace_db::rtl_trigger_shutdown();   // joins completion worker, delivers remaining events
// Now safe to close your database / flush tables
```

### What happens when callbacks are set

- **No SQLite file is created** — `get_trace_db()` lazy init is skipped
- **Kernel completions** route through the callback instead of `TraceDB::record_kernel()`
- **HIP API wrappers** route through the callback instead of `TraceDB::record_hip_api()`
- **`is_trace_ready()`** returns true for HIP wrappers even without a TraceDB, so API recording works immediately
- **Shutdown** skips `TraceDB::flush()`/`close()` — the consumer owns flushing
- **Fallback**: if no callback is set, everything works as before (standalone mode)

### Shutdown ordering

Call `rtl_trigger_shutdown()` before finalizing your own storage. This function:

1. Joins the HSA completion worker thread (waits for in-flight kernel signals)
2. Drains remaining dispatch data, delivering final events through your callback
3. Cleans up the signal pool and queue map

After it returns, no more callbacks will fire. Safe to call multiple times (idempotent).

### Build integration

Add rocm-trace-lite as a git submodule and compile its source files directly into your project:

```makefile
RTL_SRC = rocm-trace-lite/src

# Submodule sources — compiled with your project
RTL_OBJS = hsa_intercept.o hip_api_intercept.o trace_db.o

# Your source that registers callbacks
MY_OBJS += my_bridge.o

CXXFLAGS += -I$(RTL_SRC) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC
LDFLAGS  += -lhsa-runtime64 -lsqlite3 -ldl -lpthread

hsa_intercept.o: $(RTL_SRC)/hsa_intercept.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS)

hip_api_intercept.o: $(RTL_SRC)/hip_api_intercept.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS)

trace_db.o: $(RTL_SRC)/trace_db.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS)
```

Notes:
- `-DAMD_INTERNAL_BUILD` is required so `hsa_api_trace.h` resolves its includes correctly
- `trace_db.cpp` is still needed — it provides `tick()`, `next_correlation_id()`, and the callback storage
- SQLite is linked but unused when callbacks are active (no file I/O occurs)
- The submodule's `OnLoad`/`OnUnload` symbols are exported from your shared library; set `HSA_TOOLS_LIB` to point to it (or auto-set it via `dladdr` during init — see RPD example below)

### Example: RPD integration

RPD adds rocm-trace-lite as a submodule and compiles the source files directly into `librpd_tracer.so`. The entire integration is a single file:

**RtlDataSource.cpp** (~160 lines):
1. `init()` — registers callbacks with `set_kernel_event_callback()` / `set_api_event_callback()`, auto-sets `HSA_TOOLS_LIB` via `dladdr` so `runTracer.sh` works unmodified
2. `on_kernel_event()` — writes `KernelEventRecord` to RPD `OpTable` + `StringTable`
3. `on_api_event()` — writes `ApiEventRecord` to RPD `ApiTable` / `KernelApiTable` / `CopyApiTable`
4. `end()` — calls `rtl_trigger_shutdown()`, ensuring all events are delivered before Logger finalizes tables

## Acknowledgments

This project was inspired by and builds upon the work of:
- **Jeff Daily**'s [ROCm Tracer for GPU (RTG)](https://github.com/ROCm/rtg_tracer) — pioneered the HSA_TOOLS_LIB interception approach for lightweight GPU kernel tracing
- **Michael Wootton**'s [rocmProfileData (RPD)](https://github.com/ROCm/rocmProfileData) — established the SQLite-based trace format and ecosystem tools that rocm-trace-lite is compatible with

## License

MIT
