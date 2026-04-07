// GPU workload for testing rocm-trace-lite. Compiled with hipcc.
// Usage: ./gpu_workload <mode> [args...]
//   gemm <N> <iters>     — NxN FP32 GEMM, iters times
//   short <count>        — count tiny element-wise kernels
//   multi_stream <N>     — dispatch GEMM on N streams
//   multi_thread <N>     — dispatch GEMM from N host threads
//   hipgraph <replays>   — HIP graph capture + replay
//   noop                 — just init HIP, no kernels
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

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

// Multi-stream hipgraph: capture on multiple streams with cross-stream deps
// Mimics ATOM/vLLM cudagraph which captures across TP streams
static void run_hipgraph_multi_stream(int num_streams, int replays) {
    int N = 128;
    size_t bytes = N * N * sizeof(float);
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);

    // Per-stream allocations
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

    // Capture: each stream gets its own graph
    std::vector<hipGraph_t> graphs(num_streams);
    std::vector<hipGraphExec_t> graphExecs(num_streams);
    for (int s = 0; s < num_streams; s++) {
        HIP_CHECK(hipStreamBeginCapture(streams[s], hipStreamCaptureModeThreadLocal));
        for (int i = 0; i < 10; i++) {
            hipLaunchKernelGGL(matmul, grid, block, 0, streams[s],
                               dAs[s], dBs[s], dCs[s], N);
        }
        HIP_CHECK(hipStreamEndCapture(streams[s], &graphs[s]));
        HIP_CHECK(hipGraphInstantiate(&graphExecs[s], graphs[s], nullptr, nullptr, 0));
    }
    printf("Captured %d stream graphs (10 GEMM each)\n", num_streams);

    // Replay: launch all graphs concurrently
    for (int r = 0; r < replays; r++) {
        for (int s = 0; s < num_streams; s++) {
            HIP_CHECK(hipGraphLaunch(graphExecs[s], streams[s]));
        }
    }
    for (int s = 0; s < num_streams; s++) {
        HIP_CHECK(hipStreamSynchronize(streams[s]));
    }
    printf("Multi-stream graph replayed %d time(s) on %d streams\n", replays, num_streams);

    // Normal kernels after graph (mixed workload)
    for (int s = 0; s < num_streams; s++) {
        for (int i = 0; i < 3; i++) {
            hipLaunchKernelGGL(matmul, grid, block, 0, streams[s],
                               dAs[s], dBs[s], dCs[s], N);
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    printf("Normal kernels after graph: %d streams x 3 GEMM done\n", num_streams);

    for (int s = 0; s < num_streams; s++) {
        HIP_CHECK(hipGraphExecDestroy(graphExecs[s]));
        HIP_CHECK(hipGraphDestroy(graphs[s]));
        HIP_CHECK(hipStreamDestroy(streams[s]));
        (void)hipFree(dAs[s]); (void)hipFree(dBs[s]); (void)hipFree(dCs[s]);
    }
}

// Large hipgraph: many kernels in one graph (mimics ATOM's 730-address capture)
static void run_hipgraph_large(int num_kernels, int replays) {
    int N = 64;
    size_t bytes = N * N * sizeof(float);
    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemset(dA, 1, bytes));
    HIP_CHECK(hipMemset(dB, 1, bytes));

    // Also allocate vec_add buffers for mixed kernel types
    int vn = 256;
    float *vA, *vB, *vC;
    HIP_CHECK(hipMalloc(&vA, vn * sizeof(float)));
    HIP_CHECK(hipMalloc(&vB, vn * sizeof(float)));
    HIP_CHECK(hipMalloc(&vC, vn * sizeof(float)));
    HIP_CHECK(hipMemset(vA, 1, vn * sizeof(float)));
    HIP_CHECK(hipMemset(vB, 1, vn * sizeof(float)));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);

    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int i = 0; i < num_kernels; i++) {
        if (i % 5 == 0) {
            // Mix in vec_add kernels (different kernel type)
            hipLaunchKernelGGL(vec_add, dim3(1), dim3(256), 0, stream, vA, vB, vC, vn);
        } else {
            hipLaunchKernelGGL(matmul, grid, block, 0, stream, dA, dB, dC, N);
        }
    }
    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    printf("Large graph captured (%d kernels, mixed types)\n", num_kernels);

    for (int r = 0; r < replays; r++) {
        HIP_CHECK(hipGraphLaunch(graphExec, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("Large graph replayed %d time(s)\n", replays);

    // Normal kernels after graph
    for (int i = 0; i < 5; i++) {
        hipLaunchKernelGGL(matmul, grid, block, 0, stream, dA, dB, dC, N);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("Normal kernels after large graph: 5 GEMM done\n");

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
    (void)hipFree(vA); (void)hipFree(vB); (void)hipFree(vC);
}

// Multi-stream large hipgraph with interleaved replay + normal dispatch
// Closest to ATOM TP=8 cudagraph behavior
static void run_hipgraph_stress(int num_streams, int kernels_per_graph, int replays) {
    int N = 64;
    size_t bytes = N * N * sizeof(float);
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);

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

    // Phase 1: Capture per-stream graphs with many kernels
    std::vector<hipGraph_t> graphs(num_streams);
    std::vector<hipGraphExec_t> graphExecs(num_streams);
    for (int s = 0; s < num_streams; s++) {
        HIP_CHECK(hipStreamBeginCapture(streams[s], hipStreamCaptureModeThreadLocal));
        for (int i = 0; i < kernels_per_graph; i++) {
            hipLaunchKernelGGL(matmul, grid, block, 0, streams[s],
                               dAs[s], dBs[s], dCs[s], N);
        }
        HIP_CHECK(hipStreamEndCapture(streams[s], &graphs[s]));
        HIP_CHECK(hipGraphInstantiate(&graphExecs[s], graphs[s], nullptr, nullptr, 0));
    }
    printf("Stress: captured %d graphs x %d kernels\n", num_streams, kernels_per_graph);

    // Phase 2: Interleaved replay + normal dispatch (mimics inference loop)
    for (int r = 0; r < replays; r++) {
        // Graph replay on all streams
        for (int s = 0; s < num_streams; s++) {
            HIP_CHECK(hipGraphLaunch(graphExecs[s], streams[s]));
        }
        // Normal dispatch between replays (like ATOM's non-graph ops)
        for (int s = 0; s < num_streams; s++) {
            hipLaunchKernelGGL(matmul, grid, block, 0, streams[s],
                               dAs[s], dBs[s], dCs[s], N);
        }
    }
    for (int s = 0; s < num_streams; s++) {
        HIP_CHECK(hipStreamSynchronize(streams[s]));
    }
    printf("Stress: %d replays x %d streams (interleaved) done\n", replays, num_streams);

    for (int s = 0; s < num_streams; s++) {
        HIP_CHECK(hipGraphExecDestroy(graphExecs[s]));
        HIP_CHECK(hipGraphDestroy(graphs[s]));
        HIP_CHECK(hipStreamDestroy(streams[s]));
        (void)hipFree(dAs[s]); (void)hipFree(dBs[s]); (void)hipFree(dCs[s]);
    }
}

// Multi-process hipgraph: fork N processes, each captures and replays graph
// Mimics ATOM TP=8 where each worker independently does cudagraph
// NOTE: must be called BEFORE hipGetDeviceCount to avoid fork-after-HIP-init
static void run_hipgraph_multi_proc(int num_procs, int kernels_per_graph, int replays) {
    std::vector<pid_t> children;
    for (int p = 0; p < num_procs; p++) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process — fresh HIP init
            int dev_count = 0;
            HIP_CHECK(hipGetDeviceCount(&dev_count));
            HIP_CHECK(hipSetDevice(p % dev_count));
            int N = 64;
            size_t bytes = N * N * sizeof(float);
            float *dA, *dB, *dC;
            HIP_CHECK(hipMalloc(&dA, bytes));
            HIP_CHECK(hipMalloc(&dB, bytes));
            HIP_CHECK(hipMalloc(&dC, bytes));
            HIP_CHECK(hipMemset(dA, 1, bytes));
            HIP_CHECK(hipMemset(dB, 1, bytes));

            hipStream_t stream;
            HIP_CHECK(hipStreamCreate(&stream));
            dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);

            // Capture
            HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
            for (int i = 0; i < kernels_per_graph; i++) {
                hipLaunchKernelGGL(matmul, grid, block, 0, stream, dA, dB, dC, N);
            }
            hipGraph_t graph;
            HIP_CHECK(hipStreamEndCapture(stream, &graph));
            hipGraphExec_t graphExec;
            HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

            // Replay
            for (int r = 0; r < replays; r++) {
                HIP_CHECK(hipGraphLaunch(graphExec, stream));
            }
            HIP_CHECK(hipStreamSynchronize(stream));

            // Normal kernels after graph
            for (int i = 0; i < 3; i++) {
                hipLaunchKernelGGL(matmul, grid, block, 0, stream, dA, dB, dC, N);
            }
            HIP_CHECK(hipStreamSynchronize(stream));

            HIP_CHECK(hipGraphExecDestroy(graphExec));
            HIP_CHECK(hipGraphDestroy(graph));
            HIP_CHECK(hipStreamDestroy(stream));
            (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
            _exit(0);
        }
        children.push_back(pid);
    }

    // Wait for all children
    int fail = 0;
    for (pid_t pid : children) {
        int status;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "Child %d failed (status=%d)\n", pid, status);
            fail++;
        }
    }
    if (fail == 0)
        printf("Multi-proc graph: %d procs x %d kernels x %d replays — all passed\n",
               num_procs, kernels_per_graph, replays);
    else
        printf("Multi-proc graph: %d/%d processes FAILED\n", fail, num_procs);
}

static void run_hipgraph(int replays) {
    int N = 128;
    float *dA, *dB, *dC;
    size_t bytes = N * N * sizeof(float);
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemset(dA, 1, bytes));
    HIP_CHECK(hipMemset(dB, 1, bytes));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);

    // Capture phase: record a sequence of kernel launches into a graph
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int i = 0; i < 5; i++) {
        hipLaunchKernelGGL(matmul, grid, block, 0, stream, dA, dB, dC, N);
    }
    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    printf("HIP graph captured (5 GEMM kernels)\n");

    // Replay phase: launch the pre-recorded graph
    for (int r = 0; r < replays; r++) {
        HIP_CHECK(hipGraphLaunch(graphExec, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("HIP graph replayed %d time(s)\n", replays);

    // Also run some normal (non-graph) kernels to verify mixed workload
    for (int i = 0; i < 3; i++) {
        hipLaunchKernelGGL(matmul, grid, block, 0, stream, dA, dB, dC, N);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("Normal kernels after graph: 3 GEMM done\n");

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
    hipFree(dA); hipFree(dB); hipFree(dC);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mode> [args...]\n"
                "Modes:\n"
                "  gemm <N> <iters>                    — NxN FP32 GEMM\n"
                "  short <count>                       — tiny element-wise kernels\n"
                "  multi_stream <N>                    — GEMM on N streams\n"
                "  multi_thread <N>                    — GEMM from N threads\n"
                "  multi_gpu <N>                       — GEMM on N GPUs\n"
                "  multi_gpu_stream <N> <streams>      — N GPUs x streams\n"
                "  hipgraph <replays>                  — single-stream graph\n"
                "  hipgraph_ms <streams> <replays>     — multi-stream graph\n"
                "  hipgraph_large <kernels> <replays>  — large graph (many kernels)\n"
                "  hipgraph_stress <streams> <kpg> <replays> — full stress\n"
                "  hipgraph_mp <procs> <kpg> <replays>   — multi-process graph\n"
                "  noop\n", argv[0]);
        return 1;
    }

    // hipgraph_mp must fork BEFORE any HIP call to avoid fork-after-init issues
    if (strcmp(argv[1], "hipgraph_mp") == 0) {
        int np = argc > 2 ? atoi(argv[2]) : 8;
        int kpg = argc > 3 ? atoi(argv[3]) : 50;
        int rp = argc > 4 ? atoi(argv[4]) : 5;
        run_hipgraph_multi_proc(np, kpg, rp);
        return 0;
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
    } else if (strcmp(argv[1], "hipgraph") == 0) {
        int replays = argc > 2 ? atoi(argv[2]) : 3;
        run_hipgraph(replays);
    } else if (strcmp(argv[1], "hipgraph_ms") == 0) {
        int ns = argc > 2 ? atoi(argv[2]) : 4;
        int replays = argc > 3 ? atoi(argv[3]) : 3;
        run_hipgraph_multi_stream(ns, replays);
    } else if (strcmp(argv[1], "hipgraph_large") == 0) {
        int nk = argc > 2 ? atoi(argv[2]) : 100;
        int replays = argc > 3 ? atoi(argv[3]) : 3;
        run_hipgraph_large(nk, replays);
    } else if (strcmp(argv[1], "hipgraph_stress") == 0) {
        int ns = argc > 2 ? atoi(argv[2]) : 8;
        int kpg = argc > 3 ? atoi(argv[3]) : 50;
        int replays = argc > 4 ? atoi(argv[4]) : 10;
        run_hipgraph_stress(ns, kpg, replays);
    } else if (strcmp(argv[1], "noop") == 0) {
        printf("noop done\n");
    } else {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        return 1;
    }
    return 0;
}
