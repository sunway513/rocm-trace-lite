# CLI Reference

rocm-trace-lite provides the `rtl` command-line tool. (`rpd-lite` also works as an alias.)

## rtl trace

Trace a GPU workload and generate profiling output.

```bash
rtl trace [-o OUTPUT] COMMAND [ARGS...]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `trace.db` | Output trace file path |

**Output files generated:**

| File | Description |
|------|-------------|
| `trace.db` | SQLite trace database (RPD format) |
| `trace_summary.txt` | Text summary of top kernels |
| `trace.json.gz` | Compressed Perfetto JSON (open in ui.perfetto.dev) |

**Examples:**

```bash
# Basic tracing
rtl trace -o trace.db python3 my_model.py

# Multi-GPU with torchrun
rtl trace -o trace.db torchrun --nproc_per_node=4 train.py

# Trace a shell command
rtl trace -o trace.db -- ./my_hip_app --batch-size 32
```

## rtl summary

Print top kernels and GPU utilization from a trace.

```bash
rtl summary [-n LIMIT] INPUT
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --limit` | 20 | Number of top kernels to show |

**Example:**

```bash
rtl summary -n 10 trace.db
```

## rtl convert

Convert an RPD trace to Perfetto/Chrome Trace JSON.

```bash
rtl convert [-o OUTPUT] INPUT
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `<input>.json` | Output JSON file path |

**Example:**

```bash
rtl convert trace.db -o trace.json
# Open trace.json in https://ui.perfetto.dev
```

## rtl info

Show structural information about a trace file.

```bash
rtl info INPUT
```

**Example:**

```bash
rtl info trace.db
```

```text
Trace: trace.db
  Size: 1.2 MB
  Tables: rocpd_api, rocpd_metadata, rocpd_op, rocpd_string
  rocpd_op: 728 rows
  rocpd_string: 45 rows
  Duration: 13.247s
  Unique kernels: 5
```

## Environment variables

| Variable | Values | Description |
|----------|--------|-------------|
| `RTL_OUTPUT` | file path | Output trace file (alternative to `-o` flag) |
| `RTL_NO_INJECT` | `1` | Disable `hsa_amd_queue_intercept_create` and signal injection. No kernel timestamps. HIP API tracing still works. Use for CUDAGraph compatibility. |
| `RTL_DEBUG` | `1` | Log per-call summary: intercept call count, device ID, batch skip decisions |
| `RTL_DEBUG` | `2` | Log per-packet details: AQL type, signal handle, kernel object address |
