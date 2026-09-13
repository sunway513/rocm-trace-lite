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

Native GPU CI uses the smaller ROCm 10 development image pinned in
`.github/workflows/gpu-tests.yml`. Its HIP workloads, SQLite checks, and Python
test harness do not depend on PyTorch or vLLM. CI adds the SDK runtime library
path explicitly because the development image does not register HIP with the
system loader. The vLLM image above remains the serving and latency baseline.

TheRock 10.0 contains ROCR fix
[`559d48b1`](https://github.com/ROCm/rocm-systems/commit/559d48b1f013a2a8e9decd2557508de7ac6c6b10),
which sizes the intercept staging buffer to the queue rather than 256 packets.
RTL removes the old batch-skip workaround and requires this fix in every HSA profiling mode. Standard/full now both capture graph replay; lite retains only its per-dispatch signal filter. The runtime also exposes a new compatibility issue:
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
exactly 148,480 named kernels per trace plus a valid SQLite database, independently in standard and full modes. Modes run sequentially per GPU; selected GPUs run concurrently. It
returns nonzero for missing/empty/partial traces. It is a graph capture gate,
not a substitute for GPT-OSS and DeepSeek serving validation.

Before claiming serving functionality, run real requests with the allocated tensor-parallel size (TP=4 or TP=8), retain
graph execution, check all rank traces and prefill/decode windows, and verify
completed record counts and final merge integrity. Validation of the combined
multi-process/shutdown PRs must be identified separately from this PR alone.


## Serving measurements

`serving_latency.py` measures non-streaming requests. `serving_latency_stream.py`
uses SSE and reports end-to-end latency, first nonempty content, and an estimated
TPOT (the content span divided by output tokens minus one). These are client
transport measurements, not isolated GPU prefill/decode durations. Both require
fixed prompt/output lengths, a pinned server manifest, graphs enabled and prefix
caching disabled. Missing worker acknowledgements or invalid responses fail the
run before its timing can be accepted. CPU CI covers truncated streams, incorrect
length/finish metadata, and missing, duplicate or unsynchronized workers.

Example against an already allocated, task-owned TP4 server:

```bash
python3 benchmarks/e2e/serving_latency_stream.py \
  --url http://127.0.0.1:18913 --model gptoss --condition candidate-full \
  --manifest /results/server-manifest.json --output /results/requests \
  --tensor-parallel-size 4 --worker-setup-barrier \
  --prompt-tokens 709 --prompt-repetitions 24 --output-tokens 128 \
  --concurrency 1 8 --requests 32 --rounds 1
```

Tokenize the generated prompt with the selected model first; 709 is only an
example. The manifest must contain `image_digest`, `server_command`,
`model_config_sha256`, `rtl_mode`, `rtl_library_sha256`, `tensor_parallel_size`,
`graph_enabled`, and `prefix_cache_enabled`. For a development validation server,
load `rtl_e2e_worker.TraceValidationWorker` through vLLM's worker extension option
and keep its development RPC endpoint on an isolated loopback binding.

The client does not allocate GPUs or certify trace completeness. The caller must
record physical device mapping and exclusivity, verify each worker's loaded
library hash and graph replay, finalize traces, check zero lost completions,
validate per-phase device coverage and lossless merging, and release only its
own server. Use at least three independent server restarts with rotated condition
order. Compare baseline and candidate full capture on the same pinned serving
image; retain a no-profiler control to show total instrumentation overhead. Do not
pool an older ATOM/ROCm stack with vLLM/ROCm 10 as a ROCm-only comparison.
