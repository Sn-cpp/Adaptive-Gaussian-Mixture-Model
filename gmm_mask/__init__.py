from .cpu.gmm_mask_numba import GMM_Mask_Numba, warmup_mask_gmm_jit

from .cpu.gmm_mask_cpu import GMM_Mask_CPU

# CuPy and numba.cuda are optional: the CPU models must stay importable on a
# machine with no GPU, which is where the scoring harness runs. Importing
# numba.cuda succeeds without a driver and only fails when a device is first
# touched, so probe here — that way `GMM_Mask_CUDA is None` actually means
# "unavailable" instead of leaking a CudaSupportError out of a constructor.
try:
    import cupy as _cp
    # Importing cupy succeeds without a usable driver; the failure only appears
    # when a device is first touched. Probe here so `GMM_Mask_CuPy is None`
    # actually means "unavailable", instead of letting a CUDARuntimeError leak
    # out of a constructor after main.py has already promised the backend works.
    if _cp.cuda.runtime.getDeviceCount() < 1:
        raise ImportError("cupy is installed but no CUDA device is visible")
    from .gpu.gmm_mask_cupy import GMM_Mask_CuPy
except Exception:                       # pragma: no cover - cupy missing
    GMM_Mask_CuPy = None

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
