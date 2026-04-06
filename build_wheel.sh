#!/bin/bash
# Build wheel locally (requires ROCm headers installed)
set -e
echo "Building librpd_lite.so..."
make clean && make -j$(nproc)
echo "Staging .so..."
mkdir -p rocm_trace_lite/lib
cp librpd_lite.so rocm_trace_lite/lib/
echo "Building wheel..."
python3 -m build --wheel
echo "Done. Wheel in dist/"
ls -lh dist/*.whl
