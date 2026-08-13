from .cpu.gmm_em_numba import GMM_EM_Numba_CPU, warmup_em_gmm_jit

try:
    from .gpu.gmm_em_cuda import GMM_EM_CUDA, warmup_gmm_em_cuda_jit
except Exception:                       # pragma: no cover - no CUDA device
    GMM_EM_CUDA = None
    def warmup_gmm_em_cuda_jit(*a, **k):
        raise RuntimeError("GMM_EM_CUDA needs a CUDA device")
