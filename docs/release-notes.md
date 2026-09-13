# ROCm 10 candidate

This candidate adds ROCm 10 descriptor-queue interception and removes the obsolete HSA graph batch-skip workaround. Standard and full capture intercepted graph replay kernels; lite remains a partial-capture mode.

Native CI and wheel builds use the same pinned ROCm 10 development image. Source installs rebuild the native profiler and fail when it cannot be built. Candidate artifacts include a wheel, source archive, validation record and checksums. Release preparation requires an isolated wheel installation and complete graph capture on all eight MI355X GPUs.

Install the matching Linux x86_64 wheel in the supported Ubuntu 24.04 / ROCm 10 environment. See the versioned installation and performance documentation. The wheel does not include ROCm or PyTorch, and older unpatched HSA runtimes are unsupported.

Performance acceptance remains open: the dominant native slowdown has an initialization-barrier mitigation, but residual eager differences and model-serving latency require review. No fixed low-overhead promise is made. Controlled TP8 functional trace validation used explicit worker flush/finalization; abrupt termination is not a complete-trace guarantee.

This file describes a candidate, not a published production release. Refresh it against the final integrated commit and completed validation before publication.

The candidate CI wheel is built with glibc 2.39 and validated on Ubuntu 24.04.
Its current native library requires GLIBC_2.38 and cannot load on Ubuntu 22.04
(glibc 2.35), including the pinned vLLM serving image. The Linux x86_64 wheel tag
does not express that libc floor. Use a same-version sdist rebuilt inside the
target ROCm environment for older libc; this is not an expanded support claim.
Artifact `validation.json` records build libc, OS and direct native ELF GLIBC
requirements. Those direct requirements do not replace transitive dependency
and GPU validation. See the installation preflight and native-load check.
