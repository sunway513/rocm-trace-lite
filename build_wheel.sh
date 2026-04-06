#!/bin/bash
# Build wheel locally (requires ROCm headers installed)
set -e
echo "Building librtl.so..."
make clean && make -j$(nproc)
echo "Staging .so..."
mkdir -p rocm_trace_lite/lib
cp librtl.so rocm_trace_lite/lib/
echo "Building wheel..."
python3 -m build --wheel
echo "Done. Wheel in dist/"
ls -lh dist/*.whl
