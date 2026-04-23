"""L1 multi-stream HIP workload.

Per spec §4.1:
  L1-multi-stream: multi_stream 4
"""

from ._base_hip import HipWorkloadBase


class MultiStreamHip(HipWorkloadBase):
    """L1-multi-stream: GEMM dispatched on 4 streams."""

    name = "L1-multi-stream"

    def cmd(self):
        return [self._binary, "multi_stream", "4"]
