"""L1 GEMM steady-state workload — production-representative overhead measurement.

Per PR#96 overhead framing correction:
  L1-gemm-steady: gemm 64 5000 (~2.5s per run)

WHY this exists:
  L1-gemm-small (gemm 64 500, ~250ms) inflates overhead% because RTL's fixed
  startup cost (~25-40ms: HSA_TOOLS_LIB load + signal pool init + completion
  worker thread + SQLite schema init) is divided by a short run window.

  On a 2.5s run the fixed cost falls to ~1-2% of total wall — consistent with
  the 0.2-0.67% baseline Peng quoted in the Laryn 1×1 transcript and the +0.74%
  ITL measured on DSR1-MXFP4 TP=4 during PR#94 validation.

  RTL's per-kernel cost is ≤1 µs/dispatch in lite mode; the measured ms-level
  delta on short runs reflects one-time profiler initialization, not per-kernel
  overhead.
"""

from ._base_hip import HipWorkloadBase


class GemmHipSteady(HipWorkloadBase):
    """L1-gemm-steady: 64x64 GEMM x5000 iterations (~2.5s).

    Production-representative preset. Fixed startup cost (~25-40ms) is ≈1-2%
    of total wall time on this workload vs ≈10-17% on the 250ms short presets.
    """

    name = "L1-gemm-steady"

    def cmd(self):
        return [self._binary, "gemm", "64", "5000"]
