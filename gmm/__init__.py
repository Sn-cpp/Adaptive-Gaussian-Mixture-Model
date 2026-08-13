from .cpu.GMM_cpu import GMM_CPU
from .cpu.GMM_cpu_numba import GMM_CPU_NUMBA

# CuPy and numba.cuda are optional: the CPU models must stay importable on a
# machine without a GPU (macOS development).
try:
    from .gpu.GMM_cupy import GMM_CUPY
except Exception:                       # pragma: no cover - cupy missing or no GPU
    GMM_CUPY = None

try:
    # Importing numba.cuda succeeds on a machine with no driver — it only fails
    # when a device is first touched, which is inside the model constructor.
    # Probe here so `GMM_CUDA is None` actually means "unavailable" and callers
    # can say so politely instead of leaking a CudaSupportError traceback.
    from numba import cuda as _cuda
    if not _cuda.is_available():
        raise ImportError("no CUDA device")
    from .gpu.GMM_cuda import GMM_CUDA
except Exception:                       # pragma: no cover - numba.cuda missing
    GMM_CUDA = None
