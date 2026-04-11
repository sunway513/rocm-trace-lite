// repro_hipgraph_stress.hip
//
// Standalone hipGraph stress test that reproduces SEGFAULT with
// interceptible queues (hsa_amd_queue_intercept_create).
//
// The test captures a hipGraph with NUM_KERNELS kernel dispatches,
// then replays it in increasingly large batches without sync between
// replays within a batch.
//
// With a plain queue: PASS
// With interceptible queue (even pure passthrough callback): SEGFAULT
//
// Build:
//   hipcc -o repro_hipgraph_stress repro_hipgraph_stress.hip
//
// Run (should PASS):
//   ./repro_hipgraph_stress
//
// Run (should SEGFAULT with ROCm <= 7.2):
//   HSA_TOOLS_LIB=./librepro_passthrough.so ./repro_hipgraph_stress

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <setjmp.h>

// Number of kernels in the captured graph.
// 256 is the threshold where the bug typically manifests.
#ifndef NUM_KERNELS
#define NUM_KERNELS 256
#endif

// Batch sizes for replay stress test.
// The bug typically manifests after ~100-200 cumulative rapid replays.
// Escalating batch sizes to ensure we cross the threshold.
static const int BATCH_SIZES[] = {10, 20, 50, 50, 50, 50, 50, 100, 100, 100};
static const int NUM_BATCHES = sizeof(BATCH_SIZES) / sizeof(BATCH_SIZES[0]);

#define HIP_CHECK(cmd)                                                         \
    do {                                                                        \
        hipError_t err = (cmd);                                                 \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error: %s (%d) at %s:%d\n",                   \
                    hipGetErrorString(err), err, __FILE__, __LINE__);           \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Trivial kernel — just adds two vectors
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Signal handling to catch SEGFAULT and report it cleanly
static volatile sig_atomic_t g_got_segfault = 0;
static sigjmp_buf g_jmp_env;
static volatile sig_atomic_t g_jmp_set = 0;

static void segfault_handler(int sig) {
    if (g_jmp_set) {
        g_got_segfault = 1;
        siglongjmp(g_jmp_env, 1);
    }
    // If jump not set, re-raise
    signal(sig, SIG_DFL);
    raise(sig);
}

int main() {
    // Install SIGSEGV handler
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = segfault_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);

    fprintf(stderr, "=== hipGraph intercept queue stress test ===\n");
    fprintf(stderr, "NUM_KERNELS = %d\n", NUM_KERNELS);
    fprintf(stderr, "If HSA_TOOLS_LIB is set, queues will be interceptible.\n\n");

    // Check if HSA_TOOLS_LIB is set
    const char* tools_lib = getenv("HSA_TOOLS_LIB");
    if (tools_lib && tools_lib[0]) {
        fprintf(stderr, "HSA_TOOLS_LIB = %s\n", tools_lib);
    } else {
        fprintf(stderr, "HSA_TOOLS_LIB not set (using plain queues)\n");
    }

    // Allocate device memory
    const int N = 1024;
    const size_t bytes = N * sizeof(float);
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    // Initialize with some data
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

    // Create stream for graph capture
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Capture graph with NUM_KERNELS kernel dispatches
    fprintf(stderr, "\nCapturing hipGraph with %d kernel dispatches...\n", NUM_KERNELS);
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    dim3 block(256);
    dim3 grid((N + 255) / 256);
    for (int k = 0; k < NUM_KERNELS; k++) {
        vec_add<<<grid, block, 0, stream>>>(d_a, d_b, d_c, N);
    }

    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    // Instantiate the graph
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    fprintf(stderr, "Graph captured and instantiated.\n\n");

    int overall_pass = 1;

    // Replay in batches of increasing size
    for (int b = 0; b < NUM_BATCHES; b++) {
        int batch = BATCH_SIZES[b];
        fprintf(stderr, "Batch %d/%d: launching %d graph replays without sync...\n",
                b + 1, NUM_BATCHES, batch);

        g_got_segfault = 0;
        g_jmp_set = 1;

        if (sigsetjmp(g_jmp_env, 1) == 0) {
            // Normal path: launch all replays then sync
            for (int r = 0; r < batch; r++) {
                HIP_CHECK(hipGraphLaunch(graphExec, stream));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
        }

        g_jmp_set = 0;

        if (g_got_segfault) {
            fprintf(stderr, "  => FAIL: SEGFAULT during batch of %d replays!\n", batch);
            overall_pass = 0;
            break;
        } else {
            fprintf(stderr, "  => PASS (batch of %d)\n", batch);
        }
    }

    // Verify result correctness
    if (overall_pass) {
        float* h_c = (float*)malloc(bytes);
        HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));
        int correct = 1;
        for (int i = 0; i < N; i++) {
            if (h_c[i] != 3.0f) {
                fprintf(stderr, "Verification FAILED at index %d: expected 3.0, got %f\n",
                        i, h_c[i]);
                correct = 0;
                break;
            }
        }
        if (correct) {
            fprintf(stderr, "\nResult verification: PASS\n");
        }
        free(h_c);
    }

    // Cleanup
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    free(h_a);
    free(h_b);

    fprintf(stderr, "\n=== OVERALL: %s ===\n", overall_pass ? "PASS" : "FAIL");
    return overall_pass ? 0 : 1;
}
