# Changelog

## v0.3.0

### CUDAGraph / HIP graph compatibility (#67)
- **Batch skip**: Intercept callback skips signal injection for batch submissions (`count > 1`). CUDAGraph replay submits pre-recorded AQL packets that must not be modified. Injecting profiling signals corrupts the graph's execution chain and causes `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT (0x1009)`.
- **RTL_NO_INJECT=1**: Escape hatch that disables `hsa_amd_queue_intercept_create` entirely. No kernel timestamps are collected, but HIP API tracing still works. Use for full CUDAGraph compatibility when batch skip alone is insufficient (e.g., graph capture bakes stale signal handles).
- **RTL_DEBUG=1/2**: Packet-level diagnostic logging. Level 1 logs per-call summary (count, device, batch skip). Level 2 adds per-packet details (type, signal, kernel object).
- **Known limitation**: Batch skip also skips legitimate non-graph batched submissions. The HSA intercept API does not distinguish graph replay from normal multi-packet submissions.

### Signal forwarding
- Original `completion_signal` is saved before injection and forwarded via `hsa_signal_subtract_screlease` after timestamp collection. Packets with app-provided signals are now profiled correctly instead of being skipped.

### Testing
- Added `TestBatchSkip`, `TestSignalForwarding`, `TestNoInjectMode`, `TestDebugLogging` source-code audit tests
- Added `TestHipGraph` GPU E2E tests (basic, multi-stream, large, stress, batch skip logging, no-0x1009)
- HIP graph workloads added to `gpu_workload` binary (hipgraph, hipgraph_ms, hipgraph_large, hipgraph_stress)

### Bug fixes
- Fixed `profiling_set_profiler_enabled` return value not checked in RTL_NO_INJECT path
- Fixed version mismatch between `pyproject.toml` (0.2.1) and `__init__.py` (0.3.0)

## v0.2.0

### Signal injection profiling
- **Breaking**: Replaced observe-only profiling with signal injection (#31)
  - HIP runtime (ROCm 7.2) does not set completion_signal on kernel dispatch packets
  - Signal pool (64 pre-allocated, 4096 max) avoids per-dispatch allocation overhead
  - No extra HSA queues (avoids TP=8 OOM from barrier-packet approach)
- Fix: batch dispatch (`count > 1`) no longer silently dropped
- Added diagnostic counters printed at shutdown for each process

### Rename and consistency
- Renamed `librpd_lite.so` to `librtl.so`, standardized CLI on `rtl`
- All stderr messages now use `rtl:` prefix
- Added preflight diagnostics (ldd-based dependency checks)
- Kernel name demangling for readable trace output

### Documentation
- 5-tool comparison table (vs RPD, rocprofiler-sdk, roctracer, Triton Proton)
- Wheel installation instructions in README
- Simplified quick start: `rtl trace` does everything

### Testing
- 314 tests (was 130): multi-thread, multi-stream, HIP graph, multi-GPU, stress
- GPU CI on MI355X (single + 8-GPU runners)
- Pre-release validation suite with microbenchmarks and E2E
- Validated GPT-OSS 120B TP=8 on MI355X (~1M ops, 0 drops)

## v0.1.1

### Multi-process support (#28)
- Per-process trace files via `%p` PID substitution
- Automatic merge of per-process traces into single output
- GPU ID preservation across merged traces

### Packaging
- Python wheel packaging with `rtl` / `rtl` CLI tools
- `pip install rocm-trace-lite` support

### Testing
- 314 unit/integration tests (CPU + GPU)
- HIP Graph capture/replay safety
- Multi-GPU, multi-stream, multi-thread stress tests
- roctx marker integration tests

## v0.1.0

### Initial release
- HSA kernel tracing via `HSA_TOOLS_LIB` interception
- SQLite output in RPD-compatible format
- Perfetto/Chrome trace converter (`rpd2trace.py`)
- Built-in roctx shim (no libroctx64 dependency)
- Single completion worker thread (replaced thread-per-dispatch)
- Zero dependency on roctracer or rocprofiler-sdk
