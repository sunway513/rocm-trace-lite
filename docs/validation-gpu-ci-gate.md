GPU pytest failures could be hidden by `tee`: pipefail was enabled outside Docker but not in the container shell, and enough passing tests satisfied the text counter despite failures. Enable strict pipeline status inside the container and run GPU-only tests through a reusable gate that checks pytest status plus non-skipped JUnit cases. Keep the remaining suite separate so CPU successes cannot satisfy the GPU count.

- Injected regression: six passing tests plus one failure must exit nonzero.
- All-skipped regression must exit nonzero; six passing tests must succeed.
- Full CPU suite: **254 passed, 60 skipped**.
- Unit regression is collected by existing Python CI; both existing GPU jobs invoke the same checked shell gate.

The existing 8-GPU job remains optional (`continue-on-error`) by policy; this change does not silently make it a required branch-protection check. Single-GPU pytest failures are no longer masked by the pipeline.

Refs #30.
