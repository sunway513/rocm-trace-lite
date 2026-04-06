# rocm-trace-lite

Self-contained GPU kernel profiler for ROCm. **Zero roctracer/rocprofiler-sdk dependency.**

## What it does

Captures GPU kernel dispatch timestamps using only HSA runtime interception (`HSA_TOOLS_LIB`), writing to SQLite in the standard [RPD](https://github.com/ROCm/rocmProfileData) format.

| Feature | rocm-trace-lite | Original RPD |
|---------|----------------|--------------|
| Dependencies | libhsa-runtime64 + libsqlite3 | + roctracer or rocprofiler-sdk |
| GPU kernel timing | HSA signal profiling | rocprofiler callback |
| HIP API tracing | — | roctracer callback |
| roctx markers | Built-in shim | libroctx64 |
| Output format | SQLite (.db) | Same |
| Perfetto/Chrome trace | `rtl convert` | rpd2tracing.py |

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
make install    # copies librpd_lite.so to /usr/local/lib, scripts to /usr/local/bin
```

Requirements:
- ROCm (for HSA headers: `hsa/hsa.h`, `hsa/hsa_api_trace.h`)
- SQLite3 development headers (`apt install libsqlite3-dev`)
- g++ with C++17

## Quick start

```bash
rtl trace python3 my_model.py
```

Sample output:

```text
rpd_lite: lazy init, writing to trace_12345.db

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

SQLite database compatible with RPD ecosystem tools. Key tables:

- `rocpd_op` — GPU operations with start/end timestamps
- `rocpd_string` — Deduplicated string table (kernel names, op types)
- `rocpd_api` — HIP API calls (empty in lite mode)
- `rocpd_metadata` — Trace metadata

Built-in views:

```sql
-- Top kernels by total GPU time
SELECT * FROM top LIMIT 10;

-- GPU utilization per device
SELECT * FROM busy;
```

## Tests

```bash
# Run CPU-only tests (no GPU required)
make test-cpu

# GPU smoke test (requires ROCm GPU)
make test-gpu

# CI runs automatically on push via GitHub Actions
```

## License

MIT
