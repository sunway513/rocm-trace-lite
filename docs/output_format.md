# Output Format

rocm-trace-lite outputs SQLite databases compatible with the [RPD](https://github.com/ROCm/rocmProfileData) ecosystem.

## Database schema

### rocpd_op

GPU operations (kernel dispatches, roctx markers).

| Column | Type | Description |
|--------|------|-------------|
| `gpuId` | INTEGER | GPU device index (0-based). -1 for host-side markers |
| `queueId` | INTEGER | HSA queue handle |
| `sequenceId` | INTEGER | Sequence number |
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

HIP API calls (empty in rocm-trace-lite; reserved for RPD compatibility).

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
