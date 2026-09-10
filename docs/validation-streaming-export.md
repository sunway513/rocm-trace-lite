# Streaming trace export

Large traces are exported one event at a time from SQLite into plain or gzip
JSON. The converter no longer retains all source rows, parsed dispatches and
output event dictionaries simultaneously. Track metadata is retained by GPU
and hardware queue; per-dispatch queue IDs in fallback mode are traversed from
SQLite rather than stored in a Python dictionary. Python memory therefore
scales with distinct hardware tracks and one event, not with dispatch count.
SQLite temporary sorting uses disk-backed storage and a bounded page cache.

`rtl trace` now writes gzip directly through the same converter instead of
creating an uncompressed JSON and reading it into one giant bytes object.
Output is written beside the destination under a temporary name and atomically
replaced only after serialization/compression finishes. Errors preserve an
existing output and remove partial temporary files. Keep the source database;
conversion does not modify its trace contents.

Gzip uses level 3. On a 16 MiB sample from an actual GPT-OSS trace, three CPU-only
trials had median compression times of 36.7 ms (level 1), 38.2 ms (level 3) and
205.7 ms (level 9), producing 1,024,233 / 887,278 / 601,412 bytes respectively.
All decompressed bytes matched. This is a sample-specific CPU/size tradeoff,
not a model latency claim; level 3 produced about 47.5% more compressed bytes
than level 9 while taking about one fifth of its compression time.

Tests compare the packaged converter against golden output produced by the
previous implementation for hardware queues, multiple GPUs, more than 100
fallback queues, long/unicode names, host APIs and empty traces, in plain/gzip
form. Timed-event order and track metadata are preserved; host process metadata
ordering was already unspecified because it came from a set. Failure injection
checks preservation/cleanup. Separate subprocesses export 10,000 and 100,000
unique-queue dispatches under a 192 MiB address-space limit and check RSS growth.
The first local run measured 36,864 KiB peak RSS for both sizes. Results can
vary with the Python/SQLite allocator; the test allows 32 MiB growth.

To re-export an existing database without changing a running server, install
the exact candidate wheel in a separate virtualenv and use its `rtl convert`
command with a new output path. Conversion is CPU-only and does not load the
native profiler. Clear inherited `HSA_TOOLS_LIB` and `LD_PRELOAD` for that export
process. This change does not modify native sources or the capture protocol;
never replace a running workload's package as part of offline conversion.
