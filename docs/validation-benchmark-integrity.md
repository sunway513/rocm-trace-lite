The torch adapter started profiling in the parent while executing the workload in a child, allowing a successful result with no target events. Execute supported Python entrypoints inside a profiled child, enable available GPU activity, preserve workload exit status, and reject unbootstrapped in-process/native combinations. Measure peak child RSS in a fresh supervisor instead of inheriting earlier rounds or the torch-loaded parent's fork high-water mark.

Validated on **mi355-gpu-15, physical GPU 7 (MI355X)** using the freshly checked `rocm/atom` image digest `sha256:13d8564eeef3a267c1cc25410c78029097bd8f218b71ef37d32df971d73d25bb`:

- Original runner: `runner_success=True target_markers=0 gpu_kernels=0`.
- Fixed runner, same workload: `runner_success=True target_markers=2 gpu_kernels=2`.
- Combined main/benchmark CPU suite: **320 passed, 51 skipped**. Ruff checks pass for changed Python files.
- Unit tests cover high-to-low RSS, native rejection, parent-profiler rejection, child marker/PID/kernel coverage, and exit 0/nonzero trace export.
- New `Benchmark integrity` CI runs the benchmark unit suite with CPU torch and repeats the real GPU trace-content checks on the existing MI355 runner with the pinned image.

The supervisor's startup cost applies to every adapter; historical wall-time results should be rebaselined. Supported torch entrypoints are a Python script, `-m`, or `-c`; arbitrary interpreter flags/native binaries are rejected rather than compared as profiled work. RSS is the child high-water measurement, not aggregate concurrent process-tree memory. These are correctness checks, not overhead benchmarks.

Refs #97, #30.
