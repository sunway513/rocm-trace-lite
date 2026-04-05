/*
 * hip_intercept.cpp — Placeholder for HIP API interception
 *
 * NOTE: HIP API callback interception on ROCm requires roctracer's
 * hipRegisterApiCallback/hipRegisterActivityCallback, which we explicitly
 * avoid. LD_PRELOAD interception of HIP functions causes segfaults due to
 * internal re-entrancy during HIP runtime initialization.
 *
 * For kernel-level GPU profiling, the HSA queue intercept in hsa_intercept.cpp
 * captures all kernel dispatch timestamps without any roctracer dependency.
 *
 * If HIP API tracing is needed, use one of:
 *   1. Python-level torch.profiler or manual roctx markers (captured by roctx_shim)
 *   2. hipRegisterTracerCallback (HIP 5.3+ native, no roctracer needed)
 */

// Empty — all interception is done via HSA_TOOLS_LIB in hsa_intercept.cpp
