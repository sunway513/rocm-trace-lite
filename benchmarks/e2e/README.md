# ROCm 10 regression environment

Build from the repository root:

```bash
docker build -f benchmarks/e2e/Dockerfile -t rtl:rocm10 .
docker run --rm --device /dev/kfd --device /dev/dri --ipc host \
  -v "$PWD/results:/results" rtl:rocm10 \
  python3 tools/check_graph_capture.py --devices 0,1,2,3,4,5,6,7 \
  --output /results/graph
```

The base manifest is pinned to the official vLLM `nightly-rocm100` image
published on 2026-09-10. It provides ROCm SDK 10.0.0, matching PyTorch
2.12.0+rocm10.0.0 and vLLM 0.28.1rc1.dev628+g2a02f6efe.rocm100. The SDK is
installed as Python packages; use its `HIP_PATH` and `hipcc` on PATH, rather
than assuming `/opt/rocm` exists. `torch.version.hip` reports 7.15.26333 in
this image; the SDK distribution version is 10.0.0.

TheRock 10.0 contains ROCR fix
[`559d48b1`](https://github.com/ROCm/rocm-systems/commit/559d48b1f013a2a8e9decd2557508de7ac6c6b10),
which sizes the intercept staging buffer to the queue rather than 256 packets.
This enables full graph tracing, but also exposes a new compatibility issue:
CLR now creates queues through `hsa_amd_queue_create`, bypassing tools that
replace only `hsa_queue_create`. Without the descriptor hook the native graph
workload returns success while RTL records zero GPU operations.

RTL translates supported system-memory compute descriptors to interceptible
queues, retaining priority and CU mask. Other descriptors are delegated to
ROCR unchanged with an explicit incomplete-coverage warning. Device-memory
queue placement and SDMA descriptors are not yet intercepted. Do not certify
a full trace when that warning appears.

The regression executes 580 graph replays of 256 kernels on each selected
MI355X, verifies the numerical output in the native workload, and requires
exactly 148,480 named kernels per trace plus a valid SQLite database. It
returns nonzero for missing/empty/partial traces. It is a graph capture gate,
not a substitute for GPT-OSS and DeepSeek serving validation.

Before claiming serving functionality, run real requests with TP=8, retain
graph execution, check all rank traces and prefill/decode windows, and verify
completed record counts and final merge integrity. Validation of the combined
multi-process/shutdown PRs must be identified separately from this PR alone.
