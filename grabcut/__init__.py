from .cpu.raw_cpu.grabcut_cpu import GrabCut_CPU
from .cpu.numba.grabcut_numba import GrabCut_Numba, warmup_grabcut_jit

# The CUDA variants only import on a machine with a device; the CPU ones must
# stay importable without one.
try:
    from .gpu.cuda_v0.grabcut_cuda_v0 import GrabCut_CUDA_v0
except Exception:                       # pragma: no cover - no CUDA device
    GrabCut_CUDA_v0 = None

try:
    from .gpu.cuda_v1.grabcut_cuda_v1 import GrabCut_CUDA_v1, warmup_grabcut_v1_jit
except Exception:                       # pragma: no cover - no CUDA device
    GrabCut_CUDA_v1 = None
    def warmup_grabcut_v1_jit(*a, **k):
        raise RuntimeError("GrabCut_CUDA_v1 needs a CUDA device")
