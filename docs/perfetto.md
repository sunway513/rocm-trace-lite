# Perfetto Visualization

rocm-trace-lite generates [Perfetto](https://perfetto.dev/)-compatible trace files for interactive timeline visualization.

## Opening traces

1. The `rpd-lite trace` command automatically generates a compressed `trace.json.gz`
2. Open [ui.perfetto.dev](https://ui.perfetto.dev) in your browser
3. Drag and drop the `.json.gz` file (or use Open Trace File)

## Manual conversion

```bash
rpd-lite convert trace.db -o trace.json
```

## Trace structure

The Perfetto trace organizes events as:

- **Process per GPU**: Each GPU appears as a separate process (GPU 0, GPU 1, ...)
- **Thread per queue**: Hardware queues appear as threads within each GPU process
- **Complete events**: Each kernel dispatch is a "complete" event (`ph: "X"`) with start time and duration
- **Metadata events**: GPU process names and queue thread names for navigation

## Single GPU behavior

When all operations are on a single GPU, the converter uses queue-based tracks instead of GPU-based:

- If there are many unique queue IDs (> 100, typical for per-dispatch indices), they collapse to a single track
- Otherwise, each queue gets its own track for parallel stream visualization

## Multi-GPU behavior

With multiple GPUs, each GPU is a separate Perfetto process:

```
GPU 0
  ├── Queue 0x7f... (compute kernels)
  └── Queue 0x8f... (memory operations)
GPU 1
  ├── Queue 0x9f...
  └── Queue 0xaf...
```
