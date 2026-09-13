# ROCm 10 staging review

Integration branch: `staging/rocm10-rtl-integration`. Main remains at
`16f0482364d5fa30a69259c5f5f1703d13b3e85e` during this review. This branch is for
manual integrated review, not a release authorization.

## Pull request decisions

The inventory was frozen on 2026-09-13. Accepted heads are retained as ancestors
of the integrated candidate `d0119b3664fd5140795b5f2390437f09f2d57b8b`; subsequent
staging changes add the fixed CPU CI image, dependency gate and serving protocol
coverage. Closing the source PRs records consolidation into staging and does not
mean that their changes were merged into main.

| PR | Recorded head | Decision and evidence |
|---|---|---|
| #112 | `8866b2b` | Include: lossless multiprocess merge and cross-thread ROCTX; merge/native ROCTX and real HIP process tests. |
| #113 | `7ac4ecf` | Include: drain actual completions and protect admitted callbacks; deterministic dependency test and real GPU shutdown tests. |
| #114 | `ae56547` | Include: profile in the workload process and isolate RSS; measurement/runtime bundle tests and pinned Torch GPU gate. |
| #115 | `490402e` | Include: propagate pytest failures and require executed GPU tests; negative gate tests. |
| #116 | `8bdda77` | Include: ROCm 10 queue descriptor hook, fixed image and removal of graph batch skipping; exact-count graph replay on each allocated GPU. |
| #117 | `fa1f815` | Include: verified packaging and installation guidance; target-container source install and library load, plus prior installed-wheel CI evidence. |
| #118 | `0da4bb1` | Include: downloadable, attributed real TP8 excerpt with checksums. |
| #119 | `da266b4` | Include: atomic bounded-memory gzip export and optional deferred export; golden output, failure preservation and bounded-RSS tests. |
| #107 | `361e527` | Superseded by `d35bd89`: the original catches KeyboardInterrupt then references an uninitialized result. The integrated implementation waits for descendant workers, preserves raw traces on a second interrupt and returns 130; process-level CLI tests cover these cases. |
| #105 | `036d5e8` | Obsolete for this runtime baseline: it describes standard/hip graph batches as skipped. ROCm 10 now captures those batches. Retain the proposal in PR history; current docs distinguish partial lite mode and explicitly warn about unsupported descriptors. |
| #106 | `4afced3` | Defer this experiment: combines diagnostic logging with a default change to GPU-only signals and active polling, based on the old shutdown implementation. No tests qualify the new default. A follow-up would need an opt-in implementation rebased onto real-completion draining and matched CPU/GPU/trace-completeness measurements. |
| #85 | `46d3822` | Defer incompatible prototype: the pinned ROCm 10 image DOES expose the HIP profiler extension, but its Enable signature takes two arguments while the PR dynamically calls a one-argument typedef. The official header is included inside a namespace, causing a reproduced syntax-check failure with the current SDK. The header declares a beta API; long-running delivery/lifecycle and exact record coverage need separate qualification. This is distinct from the existing HIP API interception mode. |
| #84 | `43eb115` | Close the draft RFC with #85. Its design remains in PR history; adopting a new HIP CLR backend needs a revised proposal and ABI/lifecycle qualification. |

## CI changes added during consolidation

The CPU build previously used `rocm/dev-ubuntu-24.04:latest`, and its shell
`grep && exit 1 || echo Clean` could print success even after finding forbidden
dependencies. Staging pins the same ROCm 10 development image used by GPU CI and
runs `tools/check_native_dependencies.sh`. Seven executable tests cover accepted,
missing and forbidden libraries, and an `ldd` execution failure. Pushes to
`staging/**` run the CPU CI workflow.

Serving protocol tests reject incomplete SSE responses and incomplete worker
barriers. Existing GPU workflows retain actual execution-count gates; a skipped
GPU test does not count as passing GPU coverage.

## Validation environment and interpretation

The development image is
`rocm/dev-ubuntu-24.04@sha256:a90cf047f615abe70fbef83c64def0a2d549ef37a39c8ea545430aba4981b374`.
The serving image is
`vllm/vllm-openai-rocm@sha256:72e90adf360ccffb231a098a15eadfb808719c029c6cd43f6386095064552b7e`.
Only physical MI355X GPUs 4–7 are exposed to the new validation containers.
The host driver is unchanged; this qualifies ROCm 10 container userspace on this
host, not a host driver upgrade.

Source-built binaries are installed within each target image. The Ubuntu 24.04
wheel requires GLIBC 2.38 and must not be represented as compatible with the
Ubuntu 22.04 serving image's GLIBC 2.35. The serving install is rebuilt from source
inside that actual image.

Fresh initial suite: 398 passed, 17 skipped. Dedicated native GPU gate: 37 passed,
2 skipped. These suites overlap and their passing counts must not be added as
unique tests. Standard/full graph checks passed on four devices, each requiring
exactly 148,480 named kernels and valid numerical output. Additional dependency
and serving protocol tests are recorded separately after their addition.

The TP4 E2E matrix compares no profiler, pre-cleanup full (`040801f`) and candidate
full (`d0119b3`) on the identical serving image. Each model uses three independent
server restarts per condition, rotated condition order, concurrency 1/8, short and
long input, 128 fixed output tokens, plus a separate natural-stop correctness
smoke. Timing acceptance also requires worker identity, graph replay, clean GPU
allocation, zero dropped completion records, all-device phase coverage and a
lossless SQLite merge. Raw evidence lives under the owner's
`/home/pensun/rtl-staging-20260913/evidence` and `/data/rtl-staging-20260913`.

Final E2E measurements and merge readiness will be appended after the matrix
finishes. Until then, no performance-regression conclusion is claimed.
