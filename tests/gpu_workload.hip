// GPU workload for testing rocm-trace-lite. Compiled with hipcc.
// Usage: ./gpu_workload <mode> [args...]
//   gemm <N> <iters>          — NxN FP32 GEMM, iters times
//   short <count>             — count tiny element-wise kernels
//   multi_stream <N>          — dispatch GEMM on N streams
//   multi_thread <N>          — dispatch GEMM from N host threads
//   multi_gpu <N>             — dispatch GEMM on N GPUs
//   multi_gpu_stream <N> <S>  — N GPUs x S streams each
//   roctx                     — GPU work with roctx push/pop markers
//   roctx_nested              — nested roctx ranges
//   roctx_rapid <count>       — many rapid roctx marks
//   hipgraph <replays>        — HIP graph capture + replay
//   hipgraph_multi_stream     — graph capture on multiple streams
//   noop                      — just init HIP, no kernels
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <thread>
#include <vector>

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d: %s at %s:%d\n", err, hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Simple vector add kernel
__global__ void vec_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// GEMM-like kernel (not optimized, just dispatches GPU work)
__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

static void run_gemm(int N, int iters) {
    float *dA, *dB, *dC;
    size_t bytes = N * N * sizeof(float);
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemset(dA, 1, bytes));
    HIP_CHECK(hipMemset(dB, 1, bytes));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    for (int i = 0; i < iters; i++) {
        hipLaunchKernelGGL(matmul, grid, block, 0, 0, dA, dB, dC, N);
    }
    HIP_CHECK(hipDeviceSynchronize());
    printf("GEMM %dx%d x%d done\n", N, N, iters);

    hipFree(dA); hipFree(dB); hipFree(dC);
}

static void run_short_kernels(int count) {
    int n = 256;
    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, n * sizeof(float)));
    HIP_CHECK(hipMemset(dA, 1, n * sizeof(float)));
    HIP_CHECK(hipMemset(dB, 1, n * sizeof(float)));

    for (int i = 0; i < count; i++) {
        hipLaunchKernelGGL(vec_add, dim3(1), dim3(256), 0, 0, dA, dB, dC, n);
    }
    HIP_CHECK(hipDeviceSynchronize());
    printf("Short kernels x%d done\n", count);

    hipFree(dA); hipFree(dB); hipFree(dC);
}

static void run_multi_stream(int num_streams) {
    int N = 128;
    size_t bytes = N * N * sizeof(float);
    std::vector<hipStream_t> streams(num_streams);
    std::vector<float*> dAs(num_streams), dBs(num_streams), dCs(num_streams);

    for (int s = 0; s < num_streams; s++) {
        HIP_CHECK(hipStreamCreate(&streams[s]));
        HIP_CHECK(hipMalloc(&dAs[s], bytes));
        HIP_CHECK(hipMalloc(&dBs[s], bytes));
        HIP_CHECK(hipMalloc(&dCs[s], bytes));
        HIP_CHECK(hipMemset(dAs[s], 1, bytes));
        HIP_CHECK(hipMemset(dBs[s], 1, bytes));
    }

    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    for (int s = 0; s < num_streams; s++) {
        for (int i = 0; i < 10; i++) {
            hipLaunchKernelGGL(matmul, grid, block, 0, streams[s],
                               dAs[s], dBs[s], dCs[s], N);
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    printf("Multi-stream (%d streams x 10 GEMM) done\n", num_streams);

    for (int s = 0; s < num_streams; s++) {
        hipStreamDestroy(streams[s]);
        hipFree(dAs[s]); hipFree(dBs[s]); hipFree(dCs[s]);
    }
}

static void run_multi_thread(int num_threads) {
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([t]() {
            int N = 64;
            float *dA, *dB, *dC;
            size_t bytes = N * N * sizeof(float);
            HIP_CHECK(hipMalloc(&dA, bytes));
            HIP_CHECK(hipMalloc(&dB, bytes));
            HIP_CHECK(hipMalloc(&dC, bytes));
            HIP_CHECK(hipMemset(dA, 1, bytes));
            HIP_CHECK(hipMemset(dB, 1, bytes));
            dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
            for (int i = 0; i < 5; i++) {
                hipLaunchKernelGGL(matmul, grid, block, 0, 0, dA, dB, dC, N);
            }
            HIP_CHECK(hipDeviceSynchronize());
            hipFree(dA); hipFree(dB); hipFree(dC);
        });
    }
    for (auto& t : threads) t.join();
    printf("Multi-thread (%d threads x 5 GEMM) done\n", num_threads);
}

static void run_multi_gpu(int num_gpus) {
    std::vector<std::thread> threads;
    for (int g = 0; g < num_gpus; g++) {
        threads.emplace_back([g]() {
            HIP_CHECK(hipSetDevice(g));
            int N = 128;
            float *dA, *dB, *dC;
            size_t bytes = N * N * sizeof(float);
            HIP_CHECK(hipMalloc(&dA, bytes));
            HIP_CHECK(hipMalloc(&dB, bytes));
            HIP_CHECK(hipMalloc(&dC, bytes));
            HIP_CHECK(hipMemset(dA, 1, bytes));
            HIP_CHECK(hipMemset(dB, 1, bytes));
            dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
            for (int i = 0; i < 10; i++) {
                hipLaunchKernelGGL(matmul, grid, block, 0, 0, dA, dB, dC, N);
            }
            HIP_CHECK(hipDeviceSynchronize());
            (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
        });
    }
    for (auto& t : threads) t.join();
    printf("Multi-GPU (%d GPUs x 10 GEMM) done\n", num_gpus);
}

static void run_multi_gpu_stream(int num_gpus, int streams_per_gpu) {
    std::vector<std::thread> threads;
    for (int g = 0; g < num_gpus; g++) {
        threads.emplace_back([g, streams_per_gpu]() {
            HIP_CHECK(hipSetDevice(g));
            int N = 64;
            size_t bytes = N * N * sizeof(float);
            std::vector<hipStream_t> streams(streams_per_gpu);
            for (int s = 0; s < streams_per_gpu; s++) {
                HIP_CHECK(hipStreamCreate(&streams[s]));
            }
            float *dA, *dB, *dC;
            HIP_CHECK(hipMalloc(&dA, bytes));
            HIP_CHECK(hipMalloc(&dB, bytes));
            HIP_CHECK(hipMalloc(&dC, bytes));
            HIP_CHECK(hipMemset(dA, 1, bytes));
            HIP_CHECK(hipMemset(dB, 1, bytes));
            dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
            for (int s = 0; s < streams_per_gpu; s++) {
                for (int i = 0; i < 5; i++) {
                    hipLaunchKernelGGL(matmul, grid, block, 0, streams[s], dA, dB, dC, N);
                }
            }
            HIP_CHECK(hipDeviceSynchronize());
            for (int s = 0; s < streams_per_gpu; s++) {
                (void)hipStreamDestroy(streams[s]);
            }
            (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
        });
    }
    for (auto& t : threads) t.join();
    printf("Multi-GPU-Stream (%d GPUs x %d streams x 5 GEMM) done\n", num_gpus, streams_per_gpu);
}

// ---- roctx via dlopen (uses whatever HSA_TOOLS_LIB provides) ----
typedef int (*roctxPushFn)(const char*);
typedef int (*roctxPopFn)(void);
typedef void (*roctxMarkFn)(const char*);
typedef unsigned long long (*roctxStartFn)(const char*);
typedef void (*roctxStopFn)(unsigned long long);

struct RoctxAPI {
    roctxPushFn push;
    roctxPopFn pop;
    roctxMarkFn mark;
    roctxStartFn start;
    roctxStopFn stop;
    void* handle;
};

static RoctxAPI load_roctx() {
    RoctxAPI api = {};
    const char* lib = getenv("HSA_TOOLS_LIB");
    if (!lib) lib = "librtl.so";
    api.handle = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD);
    if (!api.handle) api.handle = dlopen(lib, RTLD_LAZY);
    if (!api.handle) {
        fprintf(stderr, "Warning: cannot load roctx from %s: %s\n", lib, dlerror());
        return api;
    }
    api.push = (roctxPushFn)dlsym(api.handle, "roctxRangePushA");
    api.pop = (roctxPopFn)dlsym(api.handle, "roctxRangePop");
    api.mark = (roctxMarkFn)dlsym(api.handle, "roctxMarkA");
    api.start = (roctxStartFn)dlsym(api.handle, "roctxRangeStartA");
    api.stop = (roctxStopFn)dlsym(api.handle, "roctxRangeStop");
    return api;
}

static void run_roctx() {
    auto api = load_roctx();
    if (!api.push) { fprintf(stderr, "roctx not available\n"); return; }

    int N = 128;
    float *dA, *dB, *dC;
    size_t bytes = N * N * sizeof(float);
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemset(dA, 1, bytes));
    HIP_CHECK(hipMemset(dB, 1, bytes));
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);

    api.push("outer_region");
    for (int i = 0; i < 5; i++) {
        hipLaunchKernelGGL(matmul, grid, block, 0, 0, dA, dB, dC, N);
    }
    HIP_CHECK(hipDeviceSynchronize());
    api.pop();

    api.mark("checkpoint_1");
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
    printf("roctx push/pop/mark done\n");
}

static void run_roctx_nested() {
    auto api = load_roctx();
    if (!api.push) { fprintf(stderr, "roctx not available\n"); return; }

    int N = 64;
    float *dA, *dB, *dC;
    size_t bytes = N * N * sizeof(float);
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemset(dA, 1, bytes));
    HIP_CHECK(hipMemset(dB, 1, bytes));
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);

    api.push("level_0");
    hipLaunchKernelGGL(matmul, grid, block, 0, 0, dA, dB, dC, N);
    api.push("level_1");
    hipLaunchKernelGGL(matmul, grid, block, 0, 0, dA, dB, dC, N);
    api.pop(); // level_1
    api.pop(); // level_0

    // Also test start/stop (non-nested range)
    if (api.start && api.stop) {
        auto id = api.start("range_A");
        hipLaunchKernelGGL(matmul, grid, block, 0, 0, dA, dB, dC, N);
        HIP_CHECK(hipDeviceSynchronize());
        api.stop(id);
    }

    HIP_CHECK(hipDeviceSynchronize());
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
    printf("roctx nested + start/stop done\n");
}

static void run_roctx_rapid(int count) {
    auto api = load_roctx();
    if (!api.mark) { fprintf(stderr, "roctx not available\n"); return; }
    for (int i = 0; i < count; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "mark_%d", i);
        api.mark(buf);
    }
    printf("roctx rapid %d marks done\n", count);
}

// ---- HIP Graph ----

static void run_hipgraph(int replays) {
    int N = 64;
    float *dA, *dB, *dC;
    size_t bytes = N * N * sizeof(float);
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemset(dA, 1, bytes));
    HIP_CHECK(hipMemset(dB, 1, bytes));
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Capture
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int i = 0; i < 5; i++) {
        hipLaunchKernelGGL(matmul, grid, block, 0, stream, dA, dB, dC, N);
    }
    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    hipGraphExec_t exec;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    // Replay
    for (int r = 0; r < replays; r++) {
        HIP_CHECK(hipGraphLaunch(exec, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
    (void)hipStreamDestroy(stream);
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
    printf("hipgraph capture + %d replays done\n", replays);
}

static void run_hipgraph_multi_stream() {
    int N = 64;
    float *dA, *dB, *dC;
    size_t bytes = N * N * sizeof(float);
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemset(dA, 1, bytes));
    HIP_CHECK(hipMemset(dB, 1, bytes));
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);

    // Capture on stream 0
    hipStream_t s0, s1;
    HIP_CHECK(hipStreamCreate(&s0));
    HIP_CHECK(hipStreamCreate(&s1));

    HIP_CHECK(hipStreamBeginCapture(s0, hipStreamCaptureModeGlobal));
    for (int i = 0; i < 3; i++) {
        hipLaunchKernelGGL(matmul, grid, block, 0, s0, dA, dB, dC, N);
    }
    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(s0, &graph));
    hipGraphExec_t exec;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    // Replay on stream 1
    HIP_CHECK(hipGraphLaunch(exec, s1));
    HIP_CHECK(hipStreamSynchronize(s1));

    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
    (void)hipStreamDestroy(s0);
    (void)hipStreamDestroy(s1);
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
    printf("hipgraph multi-stream done\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mode> [args...]\n", argv[0]);
        return 1;
    }

    int dev_count = 0;
    HIP_CHECK(hipGetDeviceCount(&dev_count));
    printf("HIP devices: %d\n", dev_count);
    if (dev_count == 0) {
        fprintf(stderr, "No HIP devices found\n");
        return 1;
    }

    if (strcmp(argv[1], "gemm") == 0) {
        int N = argc > 2 ? atoi(argv[2]) : 256;
        int iters = argc > 3 ? atoi(argv[3]) : 10;
        run_gemm(N, iters);
    } else if (strcmp(argv[1], "short") == 0) {
        int count = argc > 2 ? atoi(argv[2]) : 1000;
        run_short_kernels(count);
    } else if (strcmp(argv[1], "multi_stream") == 0) {
        int n = argc > 2 ? atoi(argv[2]) : 4;
        run_multi_stream(n);
    } else if (strcmp(argv[1], "multi_thread") == 0) {
        int n = argc > 2 ? atoi(argv[2]) : 4;
        run_multi_thread(n);
    } else if (strcmp(argv[1], "multi_gpu") == 0) {
        int n = argc > 2 ? atoi(argv[2]) : dev_count;
        if (n > dev_count) n = dev_count;
        run_multi_gpu(n);
    } else if (strcmp(argv[1], "multi_gpu_stream") == 0) {
        int n = argc > 2 ? atoi(argv[2]) : dev_count;
        int spc = argc > 3 ? atoi(argv[3]) : 2;
        if (n > dev_count) n = dev_count;
        run_multi_gpu_stream(n, spc);
    } else if (strcmp(argv[1], "roctx") == 0) {
        run_roctx();
    } else if (strcmp(argv[1], "roctx_nested") == 0) {
        run_roctx_nested();
    } else if (strcmp(argv[1], "roctx_rapid") == 0) {
        int count = argc > 2 ? atoi(argv[2]) : 100;
        run_roctx_rapid(count);
    } else if (strcmp(argv[1], "hipgraph") == 0) {
        int replays = argc > 2 ? atoi(argv[2]) : 1;
        run_hipgraph(replays);
    } else if (strcmp(argv[1], "hipgraph_multi_stream") == 0) {
        run_hipgraph_multi_stream();
    } else if (strcmp(argv[1], "noop") == 0) {
        printf("noop done\n");
    } else {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        return 1;
    }
    return 0;
}
