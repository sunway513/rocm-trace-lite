// Reproduce issue #67: CUDAGraph batch submission with count>=235
// Captures a HIP graph with N kernels, replays it, and verifies RTL survives.
//
// Usage: ./hipgraph_repro [num_kernels] [num_replays]
//   Default: 235 kernels, 5 replays (matches ATOM GPT-OSS CUDAGraph batch size)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d: %s at %s:%d\n", err, hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void vec_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vec_mul(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void vec_scale(float* a, float scalar, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * scalar;
}

int main(int argc, char** argv) {
    int num_kernels = argc > 1 ? atoi(argv[1]) : 235;
    int num_replays = argc > 2 ? atoi(argv[2]) : 5;

    int dev_count = 0;
    HIP_CHECK(hipGetDeviceCount(&dev_count));
    printf("HIP devices: %d\n", dev_count);
    printf("Graph capture: %d kernels, %d replays\n", num_kernels, num_replays);

    int n = 1024;
    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, n * sizeof(float)));
    HIP_CHECK(hipMemset(dA, 1, n * sizeof(float)));
    HIP_CHECK(hipMemset(dB, 1, n * sizeof(float)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Warmup — run kernels once outside graph capture
    hipLaunchKernelGGL(vec_add, dim3((n+255)/256), dim3(256), 0, stream, dA, dB, dC, n);
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("Warmup done\n");

    // Capture graph with num_kernels kernel dispatches
    hipGraph_t graph;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    for (int i = 0; i < num_kernels; i++) {
        // Mix different kernels to create a realistic graph
        switch (i % 3) {
            case 0:
                hipLaunchKernelGGL(vec_add, dim3((n+255)/256), dim3(256), 0, stream, dA, dB, dC, n);
                break;
            case 1:
                hipLaunchKernelGGL(vec_mul, dim3((n+255)/256), dim3(256), 0, stream, dA, dB, dC, n);
                break;
            case 2:
                hipLaunchKernelGGL(vec_scale, dim3((n+255)/256), dim3(256), 0, stream, dA, 2.0f, dC, n);
                break;
        }
    }

    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    printf("Graph captured: %d kernels\n", num_kernels);

    // Instantiate the graph
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    printf("Graph instantiated\n");

    // Replay — this is where the batch submission happens
    // Each replay submits all kernels as a batch (count=num_kernels in the AQL buffer)
    for (int r = 0; r < num_replays; r++) {
        HIP_CHECK(hipGraphLaunch(graphExec, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        printf("Replay %d/%d done\n", r + 1, num_replays);
    }

    printf("All %d replays completed successfully\n", num_replays);

    // Cleanup
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);

    return 0;
}
