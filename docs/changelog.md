# Changelog

## v0.2.0 (in progress)

### Signal injection profiling
- **Breaking**: Replaced observe-only profiling with signal injection (#31)
  - HIP runtime (ROCm 7.2) does not set completion_signal on kernel dispatch packets
  - Signal pool (64 pre-allocated, 4096 max) avoids per-dispatch allocation overhead
  - No extra HSA queues (avoids TP=8 OOM from barrier-packet approach)
- Fix: batch dispatch (`count > 1`) no longer silently dropped
- Added diagnostic counters printed at shutdown for each process
- Added `tests/repro_tp_multiprocess.py` and `tests/diagnose_trace.py`
- Validated TP=8 on MI355X (20,648 ops, 0 drops)

## v0.1.1

### Multi-process support (#28)
- Per-process trace files via `%p` PID substitution
- Automatic merge of per-process traces into single output
- GPU ID preservation across merged traces

### Packaging
- Python wheel packaging with `rtl` / `rtl` CLI tools
- `pip install rocm-trace-lite` support

### Testing
- 144 unit/integration tests (CPU + GPU)
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
