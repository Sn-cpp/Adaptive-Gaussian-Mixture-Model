from .cpu.gmm_mask_numba import GMM_Mask_Numba, warmup_mask_gmm_jit

from .cpu.gmm_mask_cpu import GMM_Mask_CPU

# numba.cuda is optional: the CPU models must stay importable on a
# machine with no GPU, which is where the scoring harness runs. Importing
# numba.cuda succeeds without a driver and only fails when a device is first
# touched, so probe here — that way `GMM_Mask_CUDA is None` actually means
# "unavailable" instead of leaking a CudaSupportError out of a constructor.
try:
    from numba import cuda as _cuda
    if not _cuda.is_available():
        raise ImportError("no CUDA device")
    from .gpu.gmm_mask_cuda import GMM_Mask_CUDA
    from .gpu.gmm_mask_cuda_v1 import GMM_Mask_CUDA_v1
    from .gpu.gmm_mask_cuda_v2 import GMM_Mask_CUDA_v2
except Exception:                       # pragma: no cover - no device
    GMM_Mask_CUDA = None
    GMM_Mask_CUDA_v1 = None
    GMM_Mask_CUDA_v2 = None
