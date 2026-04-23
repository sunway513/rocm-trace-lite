"""L1 short-kernels HIP workload.

Per spec §4.1:
  L1-short-kernels: short 8000
"""

from ._base_hip import HipWorkloadBase


class ShortKernelsHip(HipWorkloadBase):
    """L1-short-kernels: 8000 tiny element-wise kernels."""

    name = "L1-short-kernels"

    def cmd(self):
        return [self._binary, "short", "8000"]
