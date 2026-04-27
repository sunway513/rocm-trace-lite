#!/bin/bash
# Build the RTL hip profiler Docker image from scratch.
#
# Pre-built image available at:
#   rocm/pytorch-private:rtl-hip-profiler-v013_20260427
#
# Requirements:
#   - Docker with access to rocm/ufb-private registry
#   - Network access to GitHub (ROCm/rocm-systems)
#
# To trigger a newer ROCm nightly base image build:
#   https://github.com/ROCm/unified-framework-builds/actions/workflows/build-pytorch-docker.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TAG="${1:-rtl-hip-profiler:latest}"

echo "Building RTL hip profiler image: $TAG"
echo "  Base: rocm/ufb-private:pytorch-2.10.0-rocm7.13.0a20260424"
echo "  CLR:  ROCm/rocm-systems @ amd/dev/gandryey/ROCM-1667-12"

docker build \
  -f "$SCRIPT_DIR/Dockerfile.hip-profiler" \
  -t "$TAG" \
  "$REPO_ROOT"

echo ""
echo "Done. Test with:"
echo "  docker run --device=/dev/kfd --device=/dev/dri --group-add video \\"
echo "    --security-opt seccomp=unconfined --cap-add SYS_PTRACE --privileged \\"
echo "    --network host --ipc host -e HIP_VISIBLE_DEVICES=0 \\"
echo "    $TAG \\"
echo '    bash -c "GPU_CLR_PROFILE_OUTPUT=/dev/null rtl trace --mode hip -o /tmp/trace.db -- python3 -c \"import torch; x=torch.randn(512,512,device=\\\"cuda\\\"); torch.mm(x,x); torch.cuda.synchronize()\""'
