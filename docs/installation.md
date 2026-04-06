# Installation

## From pip (recommended)

```bash
pip install rocm-trace-lite
```

The pip package includes the pre-built `librtl.so` and CLI tools.

## From source

### Requirements

- ROCm (for HSA headers: `hsa/hsa.h`, `hsa/hsa_api_trace.h`)
- SQLite3 development headers
- g++ with C++17 support

```bash
# Install build dependencies (Ubuntu/Debian)
sudo apt install libsqlite3-dev g++

# Clone and build
git clone https://github.com/sunway513/rocm-trace-lite.git
cd rocm-trace-lite
make -j

# Install the shared library system-wide
sudo make install

# Install CLI tools
pip install -e .
```

### Verify installation

```bash
# Check the library has no forbidden dependencies
ldd librtl.so | grep -E "roctracer|rocprofiler"
# Should produce no output (clean dependency chain)

# Quick smoke test (requires GPU)
rtl trace -o test.db python3 -c "
import torch
x = torch.randn(512, 512, device='cuda')
y = x @ x
torch.cuda.synchronize()
"
rtl summary test.db
```

### Troubleshooting

If `rtl trace` reports 0 GPU ops:

1. **Check preflight output** — `rtl trace` prints diagnostic messages before
   tracing. Look for warnings about missing `libhsa-runtime64.so` or `librtl.so`.
2. **Multi-process workloads** — frameworks like ATOM/vLLM spawn GPU workers in
   subprocesses. Set env vars globally before launching:
   ```bash
   export HSA_TOOLS_LIB=$(python3 -c "from rocm_trace_lite import get_lib_path; print(get_lib_path())")
   export RPD_LITE_OUTPUT=trace_%p.db
   python3 my_model.py
   ```
3. After `make install`, run `sudo ldconfig` to update the linker cache.

## Build targets

```bash
make            # Build librtl.so
make install    # Install to /usr/local/lib and /usr/local/bin
make test-cpu   # Run non-GPU unit tests
make test-gpu   # GPU smoke test (requires ROCm GPU)
make clean      # Remove build artifacts
```

## Runtime dependencies

| Library | Source | Required |
|---------|--------|----------|
| `libhsa-runtime64.so` | ROCm runtime | Yes |
| `libsqlite3.so` | System | Yes |
| `libroctracer64.so` | **Not used** | No |
| `librocprofiler-sdk.so` | **Not used** | No |
| `libroctx64.so` | **Not used** (built-in shim) | No |
