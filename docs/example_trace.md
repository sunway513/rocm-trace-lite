# Explore a real trace before installing a model

This small example comes from the controlled GPT-OSS 120B MXFP4 run on eight MI355X GPUs. You can inspect it without a GPU, ROCm installation, or model weights.

1. Download the [Perfetto trace](./_static/examples/gptoss-tp8-excerpt.json.gz).
2. Open [ui.perfetto.dev](https://ui.perfetto.dev) and choose **Open trace file**.
3. Select the downloaded file. Look for the **GPU 0** through **GPU 7** tracks, then zoom in and select a kernel to see its duration, full name, workgroup and grid.

The [text summary](./_static/examples/gptoss-tp8-excerpt-summary.txt) is a quick preview. To explore with SQL, download the [SQLite database](./_static/examples/gptoss-tp8-excerpt.db):

```bash
sqlite3 gptoss-tp8-excerpt.db \
  'SELECT gpuId, COUNT(*) AS kernels FROM rocpd_op GROUP BY gpuId;'

sqlite3 gptoss-tp8-excerpt.db 'SELECT * FROM top LIMIT 10;'

sqlite3 gptoss-tp8-excerpt.db \
  'PRAGMA integrity_check; PRAGMA foreign_key_check;'
```

If the candidate wheel is already installed, its analysis commands work without a GPU:

```bash
rtl summary gptoss-tp8-excerpt.db
rtl convert gptoss-tp8-excerpt.db -o local-copy.json.gz
```

## What the example contains

This is a **10 ms start-time window containing 1,382 GPU kernel records across all eight GPUs**. Selection begins at the first GPU kernel after all workers emitted their E2E begin markers. Kernel durations are preserved, including kernels finishing just beyond the selection window. Counts are 173 per GPU except GPU 3 and GPU 6, which each have 172 records because the window cuts through execution.

Timestamps start at zero, hardware queue addresses are replaced by per-GPU queue numbers, and request/response text, host metadata, process IDs, HIP API records and ROCTX annotations are omitted. The source was traced in standard mode using the ROCm 10 validation stack; this is a documentation excerpt of that earlier run, not a new benchmark of the final integrated package.

The excerpt demonstrates output structure. It cannot establish model latency, profiler overhead, complete-run kernel counts or cross-GPU load balance. Consult [performance and capture limits](performance.md) for the validation scope.

The [provenance record](./_static/examples/gptoss-tp8-excerpt.provenance.json) contains the source trace checksum, selection window and counts. [SHA256SUMS](./_static/examples/SHA256SUMS) verifies the downloadable example files. Maintainers can regenerate the SQLite excerpt with:

```bash
python tools/make_trace_example.py source-trace.db excerpt.db --duration-ns 10000000
rtl convert excerpt.db -o excerpt.json.gz
```

Continue with [your first trace](quickstart.md) once your ROCm environment is ready.
