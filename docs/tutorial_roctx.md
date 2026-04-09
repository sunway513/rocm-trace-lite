# Tutorial: Profiling Prefill vs Decode with roctx Markers

rocm-trace-lite (RTL) includes a built-in roctx shim — no libroctx64 install needed. Insert markers in your code to analyze GPU kernel hotspots per inference phase.

## Quick Start

### 1. Install RTL

```bash
pip install rocm_trace_lite-0.3.2-py3-none-linux_x86_64.whl
```

### 2. Add roctx markers to your code

```python
import ctypes
import torch

# Load roctx from global symbol table (RTL injects via LD_PRELOAD)
lib = ctypes.CDLL(None)
roctx_push = lib.roctxRangePushA
roctx_push.argtypes = [ctypes.c_char_p]
roctx_push.restype = ctypes.c_int
roctx_pop = lib.roctxRangePop
roctx_pop.restype = ctypes.c_int

# Mark prefill phase
roctx_push(b"prefill")
# ... prefill code (large batch GEMM, attention, etc.) ...
torch.cuda.synchronize()
roctx_pop()

# Mark decode phase
roctx_push(b"decode")
# ... decode loop (skinny GEMM, token-by-token generation) ...
torch.cuda.synchronize()
roctx_pop()
```

### 3. Run with RTL

```bash
rtl trace -o trace.db python3 my_model.py
```

`rtl trace` automatically sets `LD_PRELOAD` so roctx symbols are globally visible to your code.

### 4. Analyze by region

```bash
rtl summary trace.db
```

Or query the SQLite trace directly:

```sql
-- List all roctx markers with duration
SELECT s.string AS marker, (o.end - o.start) / 1e6 AS duration_ms
FROM rocpd_op o JOIN rocpd_string s ON o.description_id = s.id
WHERE o.gpuId = -1 ORDER BY o.start;

-- Top kernels within a specific region
SELECT s.string AS kernel, COUNT(*) AS calls,
       ROUND(AVG(o.end - o.start) / 1e3, 1) AS avg_us
FROM rocpd_op o JOIN rocpd_string s ON o.description_id = s.id
WHERE o.gpuId >= 0 AND o.start BETWEEN <region_start> AND <region_end>
GROUP BY s.string ORDER BY SUM(o.end - o.start) DESC LIMIT 10;
```

## Example Output

```
roctx markers:
  prefill: 24.3ms
  decode:  54.9ms

=== prefill (74 ops, 0.7ms GPU time) ===
  Cijk_Ailk_Bljk (MT64x64x256)     calls=12     avg=16.1us  27.7%   <- Attention GEMM (large batch)
  Cijk_Ailk_Bljk (MT64x128x128)    calls=12     avg=13.2us  22.7%   <- FFN up-project
  Cijk_Ailk_Bljk (MT64x16x512)     calls=13     avg=12.0us  22.5%   <- FFN down-project
  vectorized_elementwise (gelu)     calls=12     avg=5.7us    9.9%   <- Activation
  ScaleAlphaVec_PostGSU8            calls=12     avg=4.8us    8.2%   <- Post-GEMM scale

=== decode (6336 ops, 36.9ms GPU time) ===
  Cijk_Ailk_Bljk (MT256x16x384)    calls=1600   avg=8.0us   34.6%   <- Skinny GEMM (batch=1)
  ScaleAlphaVec_PostGSU8_VW1        calls=1600   avg=4.5us   19.3%   <- Post-GEMM scale
  Cijk_Ailk_Bljk (MT64x16x128)     calls=768    avg=9.1us   18.9%   <- FFN GEMM
  ScaleAlphaVec_PostGSU8_VW4        calls=768    avg=4.8us   10.1%
  vectorized_elementwise (gelu)     calls=768    avg=4.0us    8.3%
```

## Key Insights

| Phase | Characteristics | Optimization direction |
|-------|----------------|----------------------|
| **Prefill** | Large batch GEMMs, few calls, long duration each | Compute-bound: optimize GEMM tile sizes |
| **Decode** | Skinny GEMMs (batch=1), many calls, short duration | Memory-bound: reduce kernel launch overhead |

## Visualize in Perfetto

```bash
rtl convert trace.db -o trace.json
# Open trace.json.gz at https://ui.perfetto.dev
```

roctx markers appear as spans on the timeline, showing prefill and decode phases with all GPU kernels nested inside.

## Notes

- Call `torch.cuda.synchronize()` before `roctx_pop()` to ensure all GPU kernels within the region have completed
- `rtl trace` automatically sets `LD_PRELOAD` for roctx support (no manual setup needed)
- No libroctx64 install needed — RTL's built-in shim provides all roctx symbols
- Markers are stored in the same SQLite trace DB alongside GPU kernel records
