# ROCm HSA Interceptible Queue SEGFAULT Reproducer

## Bug Summary

`hsa_amd_queue_intercept_create` queues cause SEGFAULT during rapid hipGraph
replay with 256+ kernel dispatches. Even a **pure passthrough callback** that
only calls `writer(in_packets, count)` without any modification triggers the
crash. Plain queues (`hsa_queue_create`) work fine under identical workloads.

## Environment

- ROCm 7.2 (also observed on 6.x)
- Any GFX9xx / CDNA / RDNA GPU
- Linux x86_64

## Files

| File | Description |
|------|-------------|
| `repro_passthrough_lib.cpp` | Minimal `HSA_TOOLS_LIB` that replaces `hsa_queue_create` with `hsa_amd_queue_intercept_create` + a zero-modification passthrough callback |
| `repro_hipgraph_stress.hip` | HIP test that captures a graph with 256 kernels and replays in batches of 10-50 |
| `Makefile` | Build and test targets |

## Build

```bash
make
```

Requires `hipcc` and `g++` from a ROCm installation. Set `ROCM_PATH` if not
at `/opt/rocm`:

```bash
make ROCM_PATH=/opt/rocm-7.2.0
```

## Reproduce

### Step 1: Verify plain queues work

```bash
make test-plain
```

Expected: all batches PASS.

### Step 2: Reproduce the crash with interceptible queues

```bash
make test-intercept
```

Expected: SEGFAULT during one of the replay batches (typically batch 3-5).

### One-liner

```bash
make test
```

## What the tool library does

The tool library (`librepro_passthrough.so`) is intentionally minimal:

1. **OnLoad**: saves original API tables, replaces `hsa_queue_create` with a
   wrapper that calls `hsa_amd_queue_intercept_create` instead.

2. **Callback**: registered via `hsa_amd_queue_intercept_register`. Does
   **only** `writer(pkts, count)` — zero packet inspection, zero signal
   injection, zero data collection. This is the absolute minimum possible
   interception.

3. **OnUnload**: prints statistics (queues created, callbacks invoked).

## Key observations

- The crash occurs even with a **pure passthrough** callback (no packet
  modification whatsoever). This rules out any tool-side bug.
- The crash threshold is approximately 256 kernel dispatches in a single graph.
  Graphs with fewer kernels typically survive.
- The crash is triggered by **rapid replay without sync** — launching multiple
  `hipGraphLaunch` calls before a single `hipStreamSynchronize`.
- Single kernel dispatches (non-graph) work fine regardless of interceptible
  queue usage.

## Workaround

Use `hsa_queue_create` (plain queues) instead of
`hsa_amd_queue_intercept_create` when graph replay workloads are expected.
Alternatively, detect graph replay (batch submissions with `count > 1` in the
intercept callback) and bypass interception for those packets.
