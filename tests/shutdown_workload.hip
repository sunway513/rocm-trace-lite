#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
__global__ void delayed_completion(int* done) {
    auto start = clock64();
    while (clock64() - start < 1000000000ULL) {}
    *done = 1;
    __threadfence_system();
}
int main() {
    auto shutdown = (void(*)())dlsym(RTLD_DEFAULT, "_ZN8trace_db20rtl_trigger_shutdownEv");
    if (!shutdown) return 2;
    int* done = nullptr;
    if (hipHostMalloc(&done, sizeof(int), hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) return 3;
    *done = 0;
    hipLaunchKernelGGL(delayed_completion, dim3(1), dim3(1), 0, 0, done);
    if (hipGetLastError() != hipSuccess) return 4;
    auto start = std::chrono::steady_clock::now();
    shutdown();
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    int observed = *(volatile int*)done;
    printf("shutdown_returned done=%d seconds=%.6f\n", observed, elapsed);
    fflush(stdout);
    // Failure exits without further HIP calls against an already finalized tracer.
    if (observed != 1) std::_Exit(5);
    if (hipDeviceSynchronize() != hipSuccess) return 6;
    return hipHostFree(done) == hipSuccess ? 0 : 7;
}
