"""GrabCut v1 — full GPU pipeline with WBPR GPU push-relabel.

Differences vs GrabCut_CUDA_v0:
  - step 9-10: cap arrays stay on device; push_relabel_gpu() runs the
    WBPR tiled-push-relabel kernel loop entirely on GPU (only small
    scalars and the height_label array cross PCIe during global relabel).
  - Morphology and Gaussian blur remain CPU (no change from v0).
"""

import numpy as np
import cv2
from numba import cuda

from grabcut.grabcut_common import GrabCut_Base
from gmm_em.gpu.gmm_em_cuda import GMM_EM_CUDA
from .push_relabel_cuda_v1 import push_relabel_gpu, warmup_push_relabel_gpu
from grabcut.gpu.cuda_v0.morphology_cuda_v0 import (
    morphological_close, morphological_open, largest_component, warmup_morph,
)
from grabcut.gpu.cuda_v0.grabcut_cuda_v0 import (
    _expf,
    _make_gc_mask_cuda,
    _calc_beta_acc_cuda,
    _calc_nweights_cuda,
    _build_tlinks_cuda,
    _build_nlinks_cuda,
    _compose_blur_cuda,
)
from settings import PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ, LAM_FACTOR


class GrabCut_CUDA_v1(GrabCut_Base):
    """GrabCut with CUDA GMMs, CUDA helper kernels, and WBPR GPU push-relabel.

    Cap arrays never leave the device during graph cut — only the final
    height_label scalar array crosses PCIe during periodic global relabeling
    (≤ N×4 B per relabeling event, not every frame).

    Device-resident intermediate state (d_*) lives on the GPU between steps;
    the only host round-trips per frame are:
      - beta accumulator scalar  (8 B download)
      - leftW / upW for lam      (2 × H×W×4 B download)
      - residual arrays for global_relabel  (6 × N×4 B, every RELABEL_FREQ iters)
      - frame copy for GaussianBlur + composite upload (H×W×3×4 B each)
    """

    def __init__(self, H: int, W: int, bg_model=None, fg_model=None):
        super().__init__(H, W,
                         bg_model or GMM_EM_CUDA(H, W),
                         fg_model or GMM_EM_CUDA(H, W))
        N = H * W
        self._BLOCK = 256
        self._grid  = (N + self._BLOCK - 1) // self._BLOCK

        # ── device scratch arrays ──────────────────────────────────────────────
        self.d_img_f32   = cuda.to_device(np.zeros((H, W, 3), dtype=np.float32))
        self.d_bg_prob   = cuda.to_device(np.zeros((H, W),    dtype=np.float32))
        self.d_gc_mask   = cuda.to_device(np.zeros((H, W),    dtype=np.uint8))
        self.d_nlp_bg    = cuda.to_device(np.zeros((H, W),    dtype=np.float32))
        self.d_nlp_fg    = cuda.to_device(np.zeros((H, W),    dtype=np.float32))
        self.d_leftW     = cuda.to_device(np.zeros((H, W),    dtype=np.float32))
        self.d_upleftW   = cuda.to_device(np.zeros((H, W),    dtype=np.float32))
        self.d_upW       = cuda.to_device(np.zeros((H, W),    dtype=np.float32))
        self.d_uprightW  = cuda.to_device(np.zeros((H, W),    dtype=np.float32))
        self.d_cap_src   = cuda.to_device(np.zeros(N,         dtype=np.float32))
        self.d_cap_snk   = cuda.to_device(np.zeros(N,         dtype=np.float32))
        self.d_cap_right = cuda.to_device(np.zeros(N,         dtype=np.float32))
        self.d_cap_down  = cuda.to_device(np.zeros(N,         dtype=np.float32))
        self.d_final_mask = cuda.to_device(np.zeros((H, W),    dtype=np.uint8))
        self.d_morph_tmp1 = cuda.to_device(np.zeros((H, W),    dtype=np.uint8))
        self.d_morph_tmp2 = cuda.to_device(np.zeros((H, W),    dtype=np.uint8))
        self.d_blurred    = cuda.to_device(np.zeros((H, W, 3), dtype=np.float32))
        self.d_composite  = cuda.to_device(np.zeros((H, W, 3), dtype=np.uint8))
        self.d_beta_acc   = cuda.to_device(np.zeros(1,         dtype=np.float64))

        self._h_beta_zero = np.zeros(1, dtype=np.float64)

    # ── public entry point ────────────────────────────────────────────────────

    def apply(self, frame, bg_prob, to_host=True, profiling=False):
        from time import perf_counter
        t0 = perf_counter()

        if hasattr(frame, 'copy_to_host'):
            d_frame = frame
        else:
            self.d_img_f32.copy_to_device(frame)
            d_frame = self.d_img_f32

        if hasattr(bg_prob, 'copy_to_host'):
            d_bg_prob = bg_prob
        else:
            self.d_bg_prob.copy_to_device(
                np.ascontiguousarray(bg_prob, dtype=np.float32))
            d_bg_prob = self.d_bg_prob

        mask, composite = self._step_kernel(d_frame, d_bg_prob)
        return mask, composite, perf_counter() - t0

    def _step_kernel(self, d_frame, d_bg_prob):
        H, W = np.int32(self.H), np.int32(self.W)

        # ── 1. GC label map ───────────────────────────────────────────────────
        _make_gc_mask_cuda[self._grid, self._BLOCK](
            d_bg_prob, self.d_gc_mask, H, W)

        # ── 2. GMM fit ────────────────────────────────────────────────────────
        self._bg_gmm.fit(d_frame, self.d_gc_mask, is_fg=False)
        self._fg_gmm.fit(d_frame, self.d_gc_mask, is_fg=True)

        # ── 3. Neg-log-prob maps ──────────────────────────────────────────────
        self._bg_gmm.neg_log_prob(d_frame, self.d_nlp_bg)
        self._fg_gmm.neg_log_prob(d_frame, self.d_nlp_fg)

        # ── 4. Beta scalar ────────────────────────────────────────────────────
        self.d_beta_acc.copy_to_device(self._h_beta_zero)
        _calc_beta_acc_cuda[self._grid, self._BLOCK](
            d_frame, self.d_beta_acc, H, W)
        cuda.synchronize()
        beta_sum = float(self.d_beta_acc.copy_to_host()[0])
        denom    = 4 * int(W) * int(H) - 3 * int(W) - 3 * int(H) + 2
        beta     = np.float32(0.0 if beta_sum <= 1e-12
                              else 1.0 / (2.0 * beta_sum / denom))

        # ── 5. N-weights ──────────────────────────────────────────────────────
        _calc_nweights_cuda[self._grid, self._BLOCK](
            d_frame, beta, np.float32(self.gamma),
            self.d_leftW, self.d_upleftW, self.d_upW, self.d_uprightW, H, W)
        cuda.synchronize()

        # ── 6. lam scalar ─────────────────────────────────────────────────────
        h_leftW = self.d_leftW.copy_to_host()
        h_upW   = self.d_upW.copy_to_host()
        max_nw  = max(float(h_leftW.max()), float(h_upW.max()))
        lam     = np.float32(max_nw * LAM_FACTOR
                             if max_nw > 0.0
                             else float(self.gamma) * LAM_FACTOR)

        # ── 7. Build t-links ──────────────────────────────────────────────────
        _build_tlinks_cuda[self._grid, self._BLOCK](
            self.d_gc_mask, self.d_nlp_bg, self.d_nlp_fg, lam,
            self.d_cap_src, self.d_cap_snk, H, W)

        # ── 8. Build n-links ──────────────────────────────────────────────────
        _build_nlinks_cuda[self._grid, self._BLOCK](
            self.d_leftW, self.d_upW,
            self.d_cap_right, self.d_cap_down, H, W)
        cuda.synchronize()

        # ── 9. Download cap arrays for GPU push_relabel ───────────────────────
        # (push_relabel_gpu accepts host arrays and uploads internally)
        h_cap_src   = self.d_cap_src.copy_to_host()
        h_cap_snk   = self.d_cap_snk.copy_to_host()
        h_cap_right = self.d_cap_right.copy_to_host()
        h_cap_down  = self.d_cap_down.copy_to_host()

        # ── 10. WBPR GPU push-relabel ─────────────────────────────────────────
        labeling = push_relabel_gpu(
            h_cap_src, h_cap_snk, h_cap_right, h_cap_down,
            H, W, PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ,
        )
        fg = (labeling == 0).reshape(int(H), int(W)).astype(np.uint8)
        np.multiply(fg, np.uint8(255), out=self._final_mask)

        # ── 11. Morphological cleanup ─────────────────────────────────────────
        morphological_close(self._final_mask, self._morph_tmp1, self._morph_tmp2,
                            H, W, radius=3)
        morphological_open(self._morph_tmp2, self._morph_tmp1, self._final_mask,
                           H, W, radius=2)
        np.copyto(self._final_mask,
                  largest_component(self._final_mask, H, W))

        # ── 12. Blurred composite ─────────────────────────────────────────────
        h_frame   = d_frame.copy_to_host()
        ks        = self.blur_ks
        h_blurred = cv2.GaussianBlur(h_frame, (ks, ks), 0)
        self.d_blurred.copy_to_device(h_blurred)
        self.d_final_mask.copy_to_device(self._final_mask)
        _compose_blur_cuda[self._grid, self._BLOCK](
            d_frame, self.d_blurred, self.d_final_mask, self.d_composite, H, W)
        cuda.synchronize()

        return self._final_mask, self.d_composite.copy_to_host()


def warmup_grabcut_v1_jit():
    """Pre-compile all kernels used by GrabCut_CUDA_v1."""
    warmup_push_relabel_gpu()
    warmup_morph()

    H, W = np.int32(4), np.int32(4)
    N    = int(H) * int(W)
    BLOCK = 256
    grid  = (N + BLOCK - 1) // BLOCK

    d_frame    = cuda.to_device(np.zeros((int(H), int(W), 3), np.float32))
    d_bg_prob  = cuda.to_device(np.zeros((int(H), int(W)),    np.float32))
    d_gc_mask  = cuda.to_device(np.zeros((int(H), int(W)),    np.uint8))
    d_nlp_bg   = cuda.to_device(np.zeros((int(H), int(W)),    np.float32))
    d_nlp_fg   = cuda.to_device(np.zeros((int(H), int(W)),    np.float32))
    d_leftW    = cuda.to_device(np.zeros((int(H), int(W)),    np.float32))
    d_upleftW  = cuda.to_device(np.zeros((int(H), int(W)),    np.float32))
    d_upW      = cuda.to_device(np.zeros((int(H), int(W)),    np.float32))
    d_uprightW = cuda.to_device(np.zeros((int(H), int(W)),    np.float32))
    d_cap_src  = cuda.to_device(np.zeros(N, np.float32))
    d_cap_snk  = cuda.to_device(np.zeros(N, np.float32))
    d_cap_right= cuda.to_device(np.zeros(N, np.float32))
    d_cap_down = cuda.to_device(np.zeros(N, np.float32))
    d_final    = cuda.to_device(np.zeros((int(H), int(W)),    np.uint8))
    d_blurred  = cuda.to_device(np.zeros((int(H), int(W), 3), np.float32))
    d_out      = cuda.to_device(np.zeros((int(H), int(W), 3), np.uint8))
    d_beta_acc = cuda.to_device(np.zeros(1, np.float64))

    _make_gc_mask_cuda  [grid, BLOCK](d_bg_prob, d_gc_mask, H, W)
    _calc_beta_acc_cuda [grid, BLOCK](d_frame, d_beta_acc, H, W)
    _calc_nweights_cuda [grid, BLOCK](d_frame, np.float32(0.1), np.float32(50.0),
                                      d_leftW, d_upleftW, d_upW, d_uprightW, H, W)
    _build_tlinks_cuda  [grid, BLOCK](d_gc_mask, d_nlp_bg, d_nlp_fg,
                                      np.float32(1.0), d_cap_src, d_cap_snk, H, W)
    _build_nlinks_cuda  [grid, BLOCK](d_leftW, d_upW, d_cap_right, d_cap_down, H, W)
    _compose_blur_cuda  [grid, BLOCK](d_frame, d_blurred, d_final, d_out, H, W)
    cuda.synchronize()
