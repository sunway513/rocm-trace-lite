"""L1 HIP-native workloads."""

from .gemm_hip import GemmHipSmall, GemmHipLarge
from .short_kernels_hip import ShortKernelsHip
from .multi_stream_hip import MultiStreamHip
from .gemm_steady_hip import GemmHipSteady

# ALL_L1_WORKLOADS: canonical list for preset enumeration and CLI workload_map checks
ALL_L1_WORKLOADS = [
    GemmHipSmall,
    GemmHipLarge,
    ShortKernelsHip,
    MultiStreamHip,
    GemmHipSteady,
]

__all__ = [
    "GemmHipSmall",
    "GemmHipLarge",
    "ShortKernelsHip",
    "MultiStreamHip",
    "GemmHipSteady",
    "ALL_L1_WORKLOADS",
]
