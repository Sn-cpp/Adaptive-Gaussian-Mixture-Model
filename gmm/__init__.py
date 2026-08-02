from .cpu.GMM_cpu import GMM_CPU
from .cpu.GMM_cpu_numba import GMM_CPU_NUMBA
from .cpu.GMM_cpu_mog2 import GMM_CPU_MOG2
from .cpu.GMM_cpu_numba_mog2 import GMM_CPU_NUMBA_MOG2

# CuPy and numba.cuda are optional: the CPU models must stay importable on a
# machine without a GPU (macOS development).
try:
    from .gpu.GMM_cupy_v0 import GMM_CUPY_V0
    from .gpu.GMM_cupy_v1 import GMM_CUPY_V1
except ImportError:                     # pragma: no cover - cupy not installed
    GMM_CUPY_V0 = GMM_CUPY_V1 = None

try:
    from .gpu.GMM_cuda_mog2 import GMM_CUDA_MOG2
except ImportError:                     # pragma: no cover - numba.cuda missing
    GMM_CUDA_MOG2 = None
