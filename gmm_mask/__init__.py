from .cpu.gmm_mask_numba import GMM_Mask_Numba, warmup_mask_gmm_jit
from .cpu.gmm_mask_cpu import GMM_Mask_CPU

# CuPy and numba.cuda are optional: on a machine without a GPU the CPU models
# must still import. Without these guards `import gmm_mask` raised
# ModuleNotFoundError and main.py could not start at all.
try:
    from .gpu.gmm_mask_cuda import GMM_Mask_CUDA
except Exception:                       # pragma: no cover - no CUDA device
    GMM_Mask_CUDA = None

try:
    from .gpu.gmm_mask_cupy import GMM_Mask_CuPy
except Exception:                       # pragma: no cover - cupy not installed
    GMM_Mask_CuPy = None
