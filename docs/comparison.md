# Comparison with Other Profilers

## Feature comparison

| Feature | rocm-trace-lite | rocprofiler-sdk | roctracer + RPD |
|---------|----------------|-----------------|-----------------|
| **Dependencies** | libhsa-runtime64 + libsqlite3 | ROCm 6.0+ full stack | roctracer + RPD |
| **GPU kernel timing** | HSA signal injection | HW counters + callbacks | roctracer callbacks |
| **HIP API tracing** | No | Yes | Yes |
| **HW performance counters** | No | Yes | No |
| **PC sampling** | No | Yes | No |
| **roctx markers** | Built-in shim | libroctx64 | libroctx64 |
| **Multi-GPU** | Automatic per-process merge | Manual | Manual |
| **Output format** | RPD SQLite + Perfetto JSON | Various | RPD SQLite |
| **Overhead** | Low (signal pool, single worker) | Medium-High | Medium |
| **New HW bring-up** | Works immediately (HSA only) | Requires rocprofiler support | Requires roctracer support |

## When to use rocm-trace-lite

- **Kernel profiling on new hardware** where rocprofiler is not yet available
- **Lightweight CI regression testing** of kernel performance
- **AI framework teams** who need clean kernel profiling without heavy dependencies
- **Quick distributed profiling** (TP>1) with automatic merge

## When to use rocprofiler-sdk instead

- You need **HW performance counters** (cache hit rates, occupancy, etc.)
- You need **PC sampling** for hotspot analysis
- You need **HIP API tracing** (API call timestamps, arguments)
- You need the full ROCm profiling ecosystem

## Relationship to RPD

rocm-trace-lite is a **standalone alternative**, not a fork of ROCm/rocmProfileData:

- Written from scratch (no RPD code dependency)
- Outputs RPD-compatible SQLite schema (interoperable with RPD tools)
- Can coexist with original RPD on the same system
