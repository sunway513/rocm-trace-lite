Shutdown previously abandoned a live GPU dispatch after a timeout, forwarded its completion, and recycled its profiling signal. Stop admitting sampled callbacks, wait for admitted producers, and drain actual completions before destroying signals. Incomplete work rotates through the queue so a later completed dependency can make progress; diagnostics print after the worker drains.

Validated on **mi355-gpu-15, physical GPU 7 (MI355X)** against main `16f0482`:

- The original library returned from shutdown after **0.104 s with GPU completion flag 0** (test exit 5).
- The fixed library passed **3/3 real GPU shutdown tests**, returning only after the kernel wrote completion flag 1.
- Full suite with the built HIP workload: **298 passed, 16 skipped**; Ruff and diff whitespace checks pass.
- A deterministic unit test executes the actual completion worker with controlled HSA signals, verifies dependent completion order, and verifies that an admitted producer prevents premature worker exit.
- New `Completion safety` CI builds/runs the native unit test and real GPU shutdown test. An obsolete string-based test requiring shutdown itself to delete queue entries is replaced by these behavioral tests.

This favors correct completion semantics over bounded shutdown when hardware never completes; it does not claim recovery from a wedged GPU. No full-mode graph or TP=8 safety claim is made. Existing graph-mode restrictions remain applicable.

Refs #19, #30.
