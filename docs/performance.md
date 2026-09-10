# Performance and capture limits

The ROCm 10 runtime changes in #116 passed native single/eight-GPU capture tests. The validation stack also passed controlled TP8 trace checks for GPT-OSS 120B MXFP4 and DeepSeek-R1-0528 FP8. **The newly packaged candidate still needs installed-wheel GPU validation, and model-serving latency acceptance is pending.** Functional trace success is not a throughput or overhead guarantee.

## Native regression investigation

On MI355X, the dominant extra eager-launch cost in the new environment follows `hipLaunchKernel → getStream → WaitActiveStreams → marker enqueue → HSA SET_EVENT`. A single device synchronization after graph warmup, outside all measured windows, removes most of the extra cost without changing the blocking stream's ordering semantics. Changing the stream to non-blocking also reduces it, but changes implicit ordering and is not a general application fix.

This is a validated initialization mitigation. The precise retained command/fence state in HIP and an upstream fix remain under investigation. See the [actual HIP source](https://github.com/ROCm/rocm-systems/blob/25e14349f2606a6b95c4ccd45eb92a1835646cf2/projects/clr/hipamd/src/hip_device.cpp) and [ROCm 10 validation PR](https://github.com/sunway513/rocm-trace-lite/pull/116).

Seven paired process rounds with the **same initialization barrier on both runtimes**, fixed kernarg/scratch/IPC settings and 100% kernel capture in every profiled timing window produced:

| Native case | Old unprofiled (µs) | New unprofiled (µs) | New standard (µs) | Paired new/old change |
|---|---:|---:|---:|---:|
| Batched eager launch | 2.246 | 2.336 | 4.399 | +3.68% |
| Eager launch + synchronization | 10.553 | 11.026 | 13.595 | +4.48% |
| Eight-kernel graph + synchronization | 28.167 | 24.763 | 47.603 | −12.08% |

Each process supplies five timing windows. The bootstrap 95% intervals for the paired new/old changes are [3.01%, 5.23%], [3.94%, 8.55%], and [−12.60%, −12.02%]. Three attempts with GPU interference were excluded and retained for audit. Old base: pinned ROCm ATOM image, ROCm 7.2.4. New base: pinned vLLM ROCm 10 image, actual HIP 7.16.26361. This does not isolate every change in the full images.

With equally complete capture, the HSA cleanup's after-full/before-full changes were +0.04%, −0.11%, and −0.07%; all confidence intervals included zero. This found no cleanup regression in these cases, rather than establishing universal equivalence.

Profiling tiny kernels has substantial relative overhead: standard/unprofiled increments were **88.23%, 23.38%, and 92.07%** here. A faster unprofiled baseline makes percentages larger. Do not apply these microbenchmark percentages to serving workloads, or advertise a universal 0%, 1%, or 2–4% cost.

## What the E2E trace validation establishes

Both model runs used eight TP workers, graph execution, fixed requests, and explicit worker synchronization/flush/finalization. The collected databases passed integrity and merge-count checks, with all eight ranks present and no reported loss. These runs establish the tested GPU/ROCTX capture path; they do not validate arbitrary forced shutdown, all HIP API calls, CPU operator tracing, tensor shapes, or every exporter.

ROCTX records remain available in SQLite but are not completely represented in the current Perfetto export. Lite intentionally filters dispatches and must not be used as a complete-trace baseline. Standard and full provide the same intercepted GPU kernel coverage on the validated runtime.

## Measuring your workload

1. Save the exact image digest, installed library hash, GPU/CPU placement, model revision, quantization, graph configuration and environment flags.
2. Use identical requests and token counts for unprofiled and profiled runs. Warm up consistently, then synchronize **every GPU worker** once before the measurement phase. A frontend-only synchronization is insufficient.
3. Keep other GPU work out of the measurement window. Repeat in independent server runs and alternate condition order; report initialization separately from steady-state latency.
4. Check errors, output lengths, all ranks, loss counters and kernel coverage before comparing latency. Save raw results and rejected attempts.
5. Compare only equivalent coverage. Fewer captured kernels can look faster while producing a less useful trace.

An old ATOM server versus a new vLLM server is a whole-stack migration comparison. It cannot by itself attribute a difference to ROCm or RTL.
