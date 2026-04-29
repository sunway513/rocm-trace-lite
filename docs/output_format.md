# Output Format

rocm-trace-lite outputs standard SQLite .db databases. The schema is compatible with [RPD](https://github.com/ROCm/rocmProfileData) ecosystem tools.

## Database schema

### rocpd_op

GPU operations (kernel dispatches, roctx markers).

| Column | Type | Description |
|--------|------|-------------|
| `gpuId` | INTEGER | GPU device index (0-based). -1 for host-side markers |
| `queueId` | INTEGER | HSA queue handle |
| `sequenceId` | INTEGER | Sequence number |
| `completionSignal` | TEXT | Dispatch info string (hwq, workgroup, grid dimensions). NULL for markers. |
| `start` | INTEGER | Start timestamp (nanoseconds) |
| `end` | INTEGER | End timestamp (nanoseconds) |
| `description_id` | INTEGER | FK to `rocpd_string` (kernel name) |
| `opType_id` | INTEGER | FK to `rocpd_string` (`KernelExecution` or `UserMarker`) |

### rocpd_string

Deduplicated string table for kernel names and operation types.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `string` | TEXT | Unique string value |

### rocpd_metadata

Trace-level metadata.

| Column | Type | Description |
|--------|------|-------------|
| `tag` | TEXT | Metadata key |
| `value` | TEXT | Metadata value |

### rocpd_api

HIP API calls. Populated when using `RTL_MODE=hip` (HIP API interception via LD_PRELOAD).

| Column | Type | Description |
|--------|------|-------------|
| `pid` | INTEGER | Process ID |
| `tid` | INTEGER | Thread ID |
| `start` | INTEGER | Start timestamp (nanoseconds) |
| `end` | INTEGER | End timestamp (nanoseconds) |
| `apiName_id` | INTEGER | FK to `rocpd_string` (API function name) |
| `args_id` | INTEGER | FK to `rocpd_string` (API arguments) |

### rocpd_api_ops

Association table linking API calls to GPU operations.

| Column | Type | Description |
|--------|------|-------------|
| `api_id` | INTEGER | FK to `rocpd_api` |
| `op_id` | INTEGER | FK to `rocpd_op` |

### rocpd_kernelapi

Kernel launch details (grid/workgroup dimensions).

| Column | Type | Description |
|--------|------|-------------|
| `api_id` | INTEGER | FK to `rocpd_api` (PK) |
| `stream` | TEXT | HIP stream handle |
| `gridX/Y/Z` | INTEGER | Grid dimensions |
| `workgroupX/Y/Z` | INTEGER | Workgroup dimensions |
| `groupSegmentSize` | INTEGER | Group segment size (bytes) |
| `privateSegmentSize` | INTEGER | Private segment size (bytes) |
| `kernelName_id` | INTEGER | FK to `rocpd_string` |

### rocpd_copyapi

Memory copy details.

| Column | Type | Description |
|--------|------|-------------|
| `api_id` | INTEGER | FK to `rocpd_api` (PK) |
| `stream` | TEXT | HIP stream handle |
| `size` | INTEGER | Copy size (bytes) |
| `dst` | TEXT | Destination address |
| `src` | TEXT | Source address |
| `kind` | INTEGER | Copy kind (H2D, D2H, D2D, etc.) |
| `sync` | INTEGER | Synchronous flag |

### rocpd_monitor

Monitoring data (reserved for future use).

| Column | Type | Description |
|--------|------|-------------|
| `deviceType` | TEXT | Device type |
| `deviceId` | INTEGER | Device ID |
| `monitorType` | TEXT | Monitor metric type |
| `start` | INTEGER | Start timestamp |
| `end` | INTEGER | End timestamp |
| `value` | TEXT | Metric value |

## Built-in views

### top

Top kernels by total GPU time.

```sql
SELECT * FROM top LIMIT 10;
```

| Column | Description |
|--------|-------------|
| `Name` | Kernel name |
| `Calls` | Number of invocations |
| `TotalNs` | Total GPU time (ns) |
| `AverageNs` | Average per-call time (ns) |
| `MinNs` | Minimum time |
| `MaxNs` | Maximum time |
| `Percentage` | % of total GPU busy time |

### busy

GPU utilization per device.

```sql
SELECT * FROM busy;
```

| Column | Description |
|--------|-------------|
| `gpuId` | GPU device index |
| `Utilization` | % of wall time with active kernels |
| `Ops` | Total kernel dispatches |
| `BusyNs` | Total GPU busy time (ns) |
| `WallNs` | Total wall time (ns) |

## Export formats

### Perfetto JSON (default)

```bash
rtl convert trace.db -o trace.json
```

Chrome Trace Event format viewable in [ui.perfetto.dev](https://ui.perfetto.dev). See [perfetto.md](perfetto.md) for details.

### rocprofv3 JSON (for TraceLens)

```bash
rtl convert trace.db --format rocprofv3 -o trace_results.json
TraceLens_generate_perf_report_rocprof --profile_json_path trace_results.json
```

Emits `rocprofiler-sdk-tool` JSON compatible with [TraceLens](https://github.com/AMD-AGI/TraceLens). Maps `rocpd_op` → `kernel_dispatch` / `memory_copy`, `rocpd_api` → `hip_api`, unique kernel names → `kernel_symbols`.

## Example queries

```sql
-- Find the slowest individual kernel execution
SELECT s.string, (o.end - o.start)/1000 as duration_us
FROM rocpd_op o
JOIN rocpd_string s ON o.description_id = s.id
ORDER BY (o.end - o.start) DESC
LIMIT 1;

-- Count kernels per GPU
SELECT gpuId, count(*) as ops
FROM rocpd_op
WHERE gpuId >= 0
GROUP BY gpuId;

-- Find NCCL/RCCL communication kernels
SELECT s.string, count(*) as calls
FROM rocpd_op o
JOIN rocpd_string s ON o.description_id = s.id
WHERE s.string LIKE '%nccl%'
GROUP BY s.string;
```
