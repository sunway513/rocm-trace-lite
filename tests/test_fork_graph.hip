/*
 * test_fork_graph.hip — Minimal reproducer for vLLM's fork + CUDA graph pattern
 *
 * Simulates:
 *   1. Parent process initializes HIP (like vLLM's API server)
 *   2. Parent forks N workers (like vLLM's multiproc_executor with fork)
 *   3. Each worker runs GPU kernels (both direct dispatches and graph replays)
 *   4. Workers exit, parent waits
 *
 * Usage:
 *   hipcc -O2 -o tests/test_fork_graph tests/test_fork_graph.hip
 *   LD_PRELOAD=./librtl.so RTL_OUTPUT=trace_%p.db RTL_MODE=full ./tests/test_fork_graph
 *   # or with HSA backend:
 *   HSA_TOOLS_LIB=./librtl.so LD_PRELOAD=./librtl.so RTL_OUTPUT=trace_%p.db RTL_MODE=full ./tests/test_fork_graph
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>

#define N 256
#define NUM_WORKERS 2
#define GRAPH_REPLAYS 5
#define DIRECT_DISPATCHES 3

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vector_scale(float* a, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= scale;
}

void worker_main(int worker_id) {
    hipSetDevice(worker_id % 8);
    fprintf(stderr, "[worker %d, pid=%d] started on GPU %d\n",
            worker_id, getpid(), worker_id % 8);

    float *dA, *dB, *dC;
    hipMalloc(&dA, N * sizeof(float));
    hipMalloc(&dB, N * sizeof(float));
    hipMalloc(&dC, N * sizeof(float));

    // Phase 1: Direct kernel dispatches (should be captured by both backends)
    for (int i = 0; i < DIRECT_DISPATCHES; i++) {
        vector_add<<<(N+255)/256, 256>>>(dA, dB, dC, N);
    }
    hipDeviceSynchronize();
    fprintf(stderr, "[worker %d] %d direct dispatches done\n", worker_id, DIRECT_DISPATCHES);

    // Phase 2: Capture a CUDA graph
    hipStream_t stream;
    hipStreamCreate(&stream);
    hipGraph_t graph;
    hipGraphExec_t graphExec;

    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    vector_add<<<(N+255)/256, 256, 0, stream>>>(dA, dB, dC, N);
    vector_scale<<<(N+255)/256, 256, 0, stream>>>(dC, 2.0f, N);
    hipStreamEndCapture(stream, &graph);
    hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    fprintf(stderr, "[worker %d] graph captured (2 kernels)\n", worker_id);

    // Phase 3: Replay the graph (should be captured by HSA queue intercept)
    for (int i = 0; i < GRAPH_REPLAYS; i++) {
        hipGraphLaunch(graphExec, stream);
    }
    hipStreamSynchronize(stream);
    fprintf(stderr, "[worker %d] %d graph replays done (expect %d kernel ops)\n",
            worker_id, GRAPH_REPLAYS, GRAPH_REPLAYS * 2);

    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);

    fprintf(stderr, "[worker %d] total expected: %d direct + %d graph = %d kernel ops\n",
            worker_id, DIRECT_DISPATCHES, GRAPH_REPLAYS * 2,
            DIRECT_DISPATCHES + GRAPH_REPLAYS * 2);
}

int main() {
    // Parent: initialize HIP (simulates vLLM API server importing torch)
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    fprintf(stderr, "[parent pid=%d] HIP initialized, %d devices\n", getpid(), device_count);

    // Fork workers
    pid_t pids[NUM_WORKERS];
    for (int i = 0; i < NUM_WORKERS; i++) {
        pids[i] = fork();
        if (pids[i] == 0) {
            worker_main(i);
            _exit(0);
        }
        fprintf(stderr, "[parent] forked worker %d as pid=%d\n", i, pids[i]);
    }

    // Wait for all workers
    for (int i = 0; i < NUM_WORKERS; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        fprintf(stderr, "[parent] worker %d (pid=%d) exited with status %d\n",
                i, pids[i], WEXITSTATUS(status));
    }

    fprintf(stderr, "[parent] all workers done\n");
    return 0;
}
