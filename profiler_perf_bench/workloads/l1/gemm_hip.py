"""L1 GEMM HIP workloads.

Per spec §4.1:
  L1-gemm-small: gemm 64 500
  L1-gemm-large: gemm 256 200
"""

from typing import Optional
from ._base_hip import HipWorkloadBase


class GemmHipSmall(HipWorkloadBase):
    """L1-gemm-small: 64x64 GEMM x500 iterations."""

    name = "L1-gemm-small"

    def cmd(self):
        return [self._binary, "gemm", "64", "500"]


class GemmHipLarge(HipWorkloadBase):
    """L1-gemm-large: 256x256 GEMM x200 iterations."""

    name = "L1-gemm-large"

    def cmd(self):
        return [self._binary, "gemm", "256", "200"]
