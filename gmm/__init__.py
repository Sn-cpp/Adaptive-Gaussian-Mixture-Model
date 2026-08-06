from .cpu.GMM_cpu import GMM_CPU
from .cpu.GMM_cpu_numba import GMM_CPU_NUMBA

# CuPy and numba.cuda are optional: the CPU models must stay importable on a
# machine without a GPU (macOS development).
try:
    from .gpu.GMM_cupy import GMM_CUPY
except Exception:                       # pragma: no cover - cupy missing or no GPU
    GMM_CUPY = None

try:
    from .gpu.GMM_cuda import GMM_CUDA
except Exception:                       # pragma: no cover - numba.cuda missing
    GMM_CUDA = None
