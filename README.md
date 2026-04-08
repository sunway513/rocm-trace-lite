# rocm-trace-lite

Self-contained GPU kernel profiler for ROCm. **Zero roctracer/rocprofiler-sdk dependency.**

## What it does

A streamlined, lightweight GPU kernel profiler. Captures dispatch timestamps using only HSA runtime interception (`HSA_TOOLS_LIB`), writing to a standard SQLite `.db` file. No dependency on HIP, roctracer, or rocprofiler-sdk.

### Comparison with other ROCm profiling tools

| Feature | rocm-trace-lite | [RPD](https://github.com/ROCm/rocmProfileData) | [rocprofiler-sdk](https://rocm.docs.amd.com/projects/rocprofiler-sdk) | [roctracer](https://rocm.docs.amd.com/projects/roctracer) | [Triton Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton) |
|---------|----------------|-----|-----------------|-----------|---------------|
| **Dependencies** | libhsa-runtime64 + libsqlite3 | + libroctracer64 | Full ROCm 6.0+ stack | libroctracer64 | libroctracer64 (AMD) |
| **GPU kernel timing** | HSA signal injection | roctracer activity | Buffered/callback tracing | Activity records | roctracer / CUPTI |
| **HIP API tracing** | — | Yes | Yes | Yes | — |
| **HSA API tracing** | — | — | Yes | Yes | — |
| **roctx markers** | Built-in shim | Via roctracer | Native | Yes (libroctx64) | Indirect |
| **HW counters** | — | — | Yes (AQLprofile) | — | NVIDIA only |
| **Output format** | SQLite (.db) | SQLite (.rpd) | CSV / JSON / Perfetto / OTF2 | Raw callbacks | JSON / Chrome Trace |
| **Perfetto visualization** | `rtl convert` | rpd2tracing.py | Native PFTrace | — | Built-in |
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
rtl trace python3 my_model.py                    # default mode
rtl trace --mode lite python3 my_model.py        # lite mode (~0% overhead)
rtl trace --mode full python3 my_model.py        # full mode (requires ROCm 7.13+)
```

### Profiling modes

| Mode | GPU timing | Graph replay | Overhead | Use case |
|------|-----------|-------------|----------|----------|
| **default** | Yes | Skipped | ~2-4% | General profiling |
| **lite** | Yes (partial) | Skipped | ~0% | Production / always-on |
| **full** | Yes (all) | Profiled | ~2-5% | Deep analysis (requires ROCm 7.13+ with [ROCR fix](https://github.com/ROCm/rocm-systems/commit/559d48b1)) |

Set via CLI (`--mode`) or env var (`RTL_MODE=lite`).

**lite** skips packets that already have a completion signal (e.g., NCCL kernels, barriers), resulting in near-zero overhead. **full** profiles everything including CUDAGraph replay batches, but requires ROCm 7.13+ to avoid a [known ROCR heap overflow](https://github.com/sunway513/rocm-trace-lite/issues/67).

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

## How it works

1. **HSA_TOOLS_LIB OnLoad** — ROCm HSA runtime calls `OnLoad()` when the library is loaded, giving us the HSA API table
2. **Queue intercept** — We replace `hsa_queue_create` to create interceptible queues via `hsa_amd_queue_intercept_create`, then register a callback on every AQL packet
3. **Kernel profiling** — For each kernel dispatch packet, we insert a profiling signal, wait for completion, then read GPU timestamps via `hsa_amd_profiling_get_dispatch_time`
4. **Symbol resolution** — We intercept `hsa_executable_freeze` to enumerate kernel symbols from code objects
5. **roctx shim** — Provides `roctxRangePushA`/`roctxRangePop`/`roctxMarkA` symbols so applications using roctx markers get captured without linking libroctx64

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

## Acknowledgments

This project was inspired by and builds upon the work of:
- **Jeff Daily**'s [ROCm Tracer for GPU (RTG)](https://github.com/ROCm/rtg_tracer) — pioneered the HSA_TOOLS_LIB interception approach for lightweight GPU kernel tracing
- **Michael Wootton**'s [rocmProfileData (RPD)](https://github.com/ROCm/rocmProfileData) — established the SQLite-based trace format and ecosystem tools that rocm-trace-lite is compatible with

## License

MIT
