Multi-process traces lose non-base HIP API rows and ROCTX IDs; process ranges also disappear when stopped on another thread. This preserves complete tables with reference/range-ID remapping, reads committed WAL data via SQLite backup, and aborts an incomplete merge without deleting its inputs. Process ROCTX ranges now use synchronized process storage, and nested Pop returns its actual nesting level.

Validated on **mi355-gpu-15, physical GPU 7 (MI355X)** against main `16f0482`:

- Before: two HIP processes each recorded 3 APIs and 11 nonzero-ROCTX records; merged output retained only 3 APIs and 11 such records. Cross-thread ranges were absent.
- After: the real-HIP regression preserves both processes, distinct range IDs, and both cross-thread markers.
- Full suite: **266 passed, 51 skipped**. Targeted native/merge/GPU suite: **6 passed**. Ruff and diff whitespace checks pass.
- Unit tests cover all linked tables, API-only processes, legacy schemas, WAL, corrupted-input recovery, cross-thread ROCTX and nesting. New `Trace integrity` CI runs native CPU tests and real HIP merge validation on the existing MI355 runner.

The changed corrupted-source test now requires an explicit failure and retained inputs instead of accepting silently incomplete output. The merge API retains source files; the CLI removes them only after successful merge. No TP=8 or performance-overhead claim is made.

Refs #30.
