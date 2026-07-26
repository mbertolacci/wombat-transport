"""Optional CUDA execution support.

Importing this package does not import CuPy or initialize a CUDA device.
"""

from wombat_transport.cuda.runtime import (
    CudaDeviceInfo,
    CudaRuntime,
    CudaUnavailableError,
    TransferStats,
)

__all__ = [
    "CudaDeviceInfo",
    "CudaRuntime",
    "CudaUnavailableError",
    "TransferStats",
]
