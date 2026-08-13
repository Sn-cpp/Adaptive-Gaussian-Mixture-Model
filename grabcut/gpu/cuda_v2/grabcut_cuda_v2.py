"""GrabCut v2 — tile-wave GPU push-relabel + full GPU morphology/blur.

Identical to GrabCut_CUDA_v1 except push_relabel_wave() is used instead of
push_relabel_tiled().  The only observable difference is that global relabeling
runs entirely on the GPU (no residual-array PCIe round-trip), saving ≈ 0.74 ms
per relabeling event and eliminating the blocking CPU sync point.
"""

import numpy as np
from numba import cuda

from grabcut.grabcut_common import GrabCut_Base
from gmm_em.gpu.gmm_em_cuda import GMM_EM_CUDA
from grabcut.gpu.cuda_v0.grabcut_cuda_v0 import (
    _make_gc_mask_cuda,
    _calc_beta_acc_cuda,
    _calc_nweights_cuda,
    _build_tlinks_cuda,
    _build_nlinks_cuda,
    _compose_blur_cuda,
)
from grabcut.gpu.morphology_gpu import (
    morphological_close, morphological_open,
    largest_component, gaussian_blur_f32,
    warmup_morph_gpu,
)
from grabcut.gpu.cuda_v1.grabcut_cuda_v1 import _max_reduce_kernel
from .push_relabel_cuda_v2 import (
    push_relabel_wave, warmup_push_relabel_wave,
)
from settings import PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ, LAM_FACTOR
from time import perf_counter

class GrabCut_CUDA_v2(GrabCut_Base):
    """GrabCut with tile-wave GPU push-relabel and full GPU morphology + blur.

    Host ↔ device transfers per frame:
      H2D  frame upload          3.7 MB  (if host input)
      D2H  beta scalar           8 B
      D2H  lam scalar            4 B
      D2H  final_mask for CC     0.3 MB
      H2D  CC mask upload        0.3 MB
      D2H  cap_right+down+snk    3×N×4 B  for residual init (once per step)
      D2H  height labels only    (N+2)×4 B  at end of push_relabel_wave()
      D2H  mask + composite      1.2 MB
      (no residual download for global relabeling — GPU BFS runs on-device)
    """

    def __init__(self, H: int, W: int, bg_model=None, fg_model=None):
        super().__init__(H, W,
                         bg_model or GMM_EM_CUDA(H, W),
                         fg_model or GMM_EM_CUDA(H, W))
        N = H * W
        self._BLOCK = 256
        self._grid  = (N + self._BLOCK - 1) // self._BLOCK

        self.d_img_f32   = cuda.to_device(np.zeros((H, W, 3), np.float32))
        self.d_bg_prob   = cuda.to_device(np.zeros((H, W),    np.float32))
        self.d_gc_mask   = cuda.to_device(np.zeros((H, W),    np.uint8))
        self.d_nlp_bg    = cuda.to_device(np.zeros((H, W),    np.float32))
        self.d_nlp_fg    = cuda.to_device(np.zeros((H, W),    np.float32))
        self.d_leftW     = cuda.to_device(np.zeros((H, W),    np.float32))
        self.d_upleftW   = cuda.to_device(np.zeros((H, W),    np.float32))
        self.d_upW       = cuda.to_device(np.zeros((H, W),    np.float32))
        self.d_uprightW  = cuda.to_device(np.zeros((H, W),    np.float32))
        self.d_cap_src   = cuda.to_device(np.zeros(N,         np.float32))
        self.d_cap_snk   = cuda.to_device(np.zeros(N,         np.float32))
        self.d_cap_right = cuda.to_device(np.zeros(N,         np.float32))
        self.d_cap_down  = cuda.to_device(np.zeros(N,         np.float32))
        self.d_final_mask = cuda.to_device(np.zeros((H, W),    np.uint8))
        self.d_morph_tmp  = cuda.to_device(np.zeros((H, W),    np.uint8))
        self.d_blur_tmp   = cuda.to_device(np.zeros((H, W, 3), np.float32))
        self.d_blurred    = cuda.to_device(np.zeros((H, W, 3), np.float32))
        self.d_composite  = cuda.to_device(np.zeros((H, W, 3), np.uint8))
        self.d_beta_acc   = cuda.to_device(np.zeros(1,         np.float64))
        self.d_lam_acc    = cuda.to_device(np.zeros(1,         np.float32))

        self._h_beta_zero = np.zeros(1, np.float64)
        self._h_lam_zero  = np.zeros(1, np.float32)

    def _step_kernel(self, frame, bg_prob, to_host, profiling):
        d_frame = frame if hasattr(frame, 'copy_to_host') else cuda.to_device(frame)
        d_bg_prob = bg_prob if hasattr(bg_prob, 'copy_to_host') else cuda.to_device(np.ascontiguousarray(bg_prob, dtype=np.float32)) 

        t0 = perf_counter()
        self._step_device(d_frame, d_bg_prob, profiling)
        cuda.synchronize()
        t1 = perf_counter()

        print("Step ", (t1-t0)*1000)

        if to_host:
            self._composite = self.d_composite.copy_to_host()
            self._final_mask = self.d_final_mask.copy_to_host()
            return self._final_mask, self._composite
        else:
            return self.d_final_mask, self.d_composite

    def _step_device(self, d_frame, d_bg_prob, profiling):
        H, W = np.int32(self.H), np.int32(self.W)

        # gc mask ~ 0.3ms
        _make_gc_mask_cuda[self._grid, self._BLOCK](
            d_bg_prob, self.d_gc_mask, H, W)

        # gmms fit + nlp ~ 13ms
        self._bg_gmm.fit(d_frame, self.d_gc_mask, is_fg=False)
        self._fg_gmm.fit(d_frame, self.d_gc_mask, is_fg=True)
        self._bg_gmm.neg_log_prob(d_frame, self.d_nlp_bg)
        self._fg_gmm.neg_log_prob(d_frame, self.d_nlp_fg)

        # self.d_beta_acc.copy_to_device(self._h_beta_zero)
        # beta compute ~ 1ms
        _calc_beta_acc_cuda[self._grid, self._BLOCK](
            d_frame, self.d_beta_acc, H, W)
        cuda.synchronize()
        beta = self.d_beta_acc.copy_to_host()[0]

        # nweights ~ 0.4ms
        _calc_nweights_cuda[self._grid, self._BLOCK](
            d_frame, beta, np.float32(self.gamma),
            self.d_leftW, self.d_upleftW, self.d_upW, self.d_uprightW, H, W)

        # lam accumulation ~ 1ms
        self.d_lam_acc.copy_to_device(self._h_lam_zero)
        N = int(H) * int(W)
        _max_reduce_kernel[self._grid, self._BLOCK](
            self.d_leftW.reshape(N), self.d_lam_acc, N)
        _max_reduce_kernel[self._grid, self._BLOCK](
            self.d_upW.reshape(N), self.d_lam_acc, N)
        cuda.synchronize()
        max_nw = self.d_lam_acc.copy_to_host()[0]
        lam    = np.float32(max_nw * LAM_FACTOR
                            if max_nw > 0.0
                            else float(self.gamma) * LAM_FACTOR)
        
        # Tlinks and Nlinks ~ 0.5ms
        _build_tlinks_cuda[self._grid, self._BLOCK](
            self.d_gc_mask, self.d_nlp_bg, self.d_nlp_fg, lam,
            self.d_cap_src, self.d_cap_snk, H, W)

        _build_nlinks_cuda[self._grid, self._BLOCK](
            self.d_leftW, self.d_upW,
            self.d_cap_right, self.d_cap_down, H, W)
        cuda.synchronize()


        # Tile-wave GPU push-relabel ~ 195ms
        self.d_final_mask = push_relabel_wave(
            self.d_cap_src, self.d_cap_snk, self.d_cap_right, self.d_cap_down,
            H, W, PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ)

        # Morphology close/open ~ 0.7ms
        morphological_close(
            self.d_final_mask, self.d_morph_tmp, self.d_final_mask,
            H, W, radius=3)
        morphological_open(
            self.d_final_mask, self.d_morph_tmp, self.d_final_mask,
            H, W, radius=2)
        cuda.synchronize()

        # Largest component(CPU BFS) ~ 6ms
        largest_component(self.d_final_mask, H, W)

        # Gaussian blur ~ 0.4ms
        gaussian_blur_f32(
            d_frame, self.d_blur_tmp, self.d_blurred,
            H, W, ksize=self.blur_ks, sigma=5.0)

        # Composite build ~ 0.1ms
        _compose_blur_cuda[self._grid, self._BLOCK](
            d_frame, self.d_blurred, self.d_final_mask, self.d_composite, H, W)


def warmup_grabcut_v2_jit():
    """Pre-compile all kernels used by GrabCut_CUDA_v2."""
    warmup_push_relabel_wave()
    warmup_morph_gpu()
