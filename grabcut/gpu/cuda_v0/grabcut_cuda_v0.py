import numpy as np
import cv2
from numba import cuda

from grabcut.grabcut_common import GrabCut_Base
from gmm_em.gpu.gmm_em_cuda import GMM_EM_CUDA
from .push_relabel_cuda_v0 import push_relabel, warmup_push_relabel
from .morphology_cuda_v0 import (
    morphological_close, morphological_open, largest_component, warmup_morph,
)
from settings import PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ, LAM_FACTOR

from time import perf_counter
from utils import line_measurer_cuda, line_measurer

class GrabCut_CUDA_v0(GrabCut_Base):
    """GrabCut with CUDA GMMs, CUDA helper kernels, and CPU push-relabel.

    _step_kernel(d_frame, d_bg_prob) accepts device arrays.  apply() accepts
    either host or device arrays and uploads as needed.

    Device-resident intermediate state (d_*) lives on the GPU between steps;
    the only host round-trips per frame are:
      - beta accumulator scalar  (8 B download)
      - leftW / upW for lam      (2 × H×W×4 B download)
      - cap_src/snk/right/down   (4 × N×4 B download, for CPU push_relabel)
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
        self.d_beta_acc   = cuda.to_device(np.zeros(1,         dtype=np.float32))

        # Pre-allocated zero arrays for device resets (avoid per-frame alloc)
        self._h_beta_zero = np.zeros(1, dtype=np.float32)

    
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
            return self._final_mask, self._composite
        else:
            return self.d_final_mask, self.d_composite

    def _step_device(self, d_frame, d_bg_prob, profiling):
        """Full GrabCut pipeline.  Both inputs must be device arrays.

        d_frame   : (H, W, 3) float32  device
        d_bg_prob : (H, W)    float32  device
        Returns   : (final_mask, composite) as host uint8 arrays.
        """
        H, W = np.int32(self.H), np.int32(self.W)

    

        # ── 1. GC label map ───────────────────────────────────────────────────

        _make_gc_mask_cuda[self._grid, self._BLOCK](
            d_bg_prob, self.d_gc_mask, H, W)

        # ── 2. GMM fit (E + M-step, GPU) ──────────────────────────────────────
        self._bg_gmm.fit(d_frame, self.d_gc_mask, is_fg=False)
        self._fg_gmm.fit(d_frame, self.d_gc_mask, is_fg=True)
        
        # ── 3. Neg-log-prob maps (GPU) ────────────────────────────────────────
        self._bg_gmm.neg_log_prob(d_frame, self.d_nlp_bg)
        self._fg_gmm.neg_log_prob(d_frame, self.d_nlp_fg)

        # ── 4. Beta scalar (GPU reduction → 1 scalar download) ────────────────
        self.d_beta_acc.copy_to_device(self._h_beta_zero)

        _calc_beta_acc_cuda[self._grid, self._BLOCK](
            d_frame, self.d_beta_acc, H, W)
        cuda.synchronize()

        beta = self.d_beta_acc.copy_to_host()[0]

        # ── 5. N-weights (GPU) ────────────────────────────────────────────────
        _calc_nweights_cuda[self._grid, self._BLOCK](
            d_frame, beta, np.float32(self.gamma),
            self.d_leftW, self.d_upleftW, self.d_upW, self.d_uprightW, H, W)
        cuda.synchronize()

        # ── 6. lam scalar (download leftW + upW for max) ──────────────────────
        h_leftW = self.d_leftW.copy_to_host()
        h_upW   = self.d_upW.copy_to_host()
        max_nw  = max(float(h_leftW.max()), float(h_upW.max()))
        lam     = np.float32(max_nw * LAM_FACTOR
                             if max_nw > 0.0
                             else float(self.gamma) * LAM_FACTOR)

        # ── 7. Build t-links (GPU) ────────────────────────────────────────────
        _build_tlinks_cuda[self._grid, self._BLOCK](
            self.d_gc_mask, self.d_nlp_bg, self.d_nlp_fg, lam,
            self.d_cap_src, self.d_cap_snk, H, W)

        # ── 8. Build n-links (GPU) ────────────────────────────────────────────
        _build_nlinks_cuda[self._grid, self._BLOCK](
            self.d_leftW, self.d_upW,
            self.d_cap_right, self.d_cap_down, H, W)
        cuda.synchronize()

        # ── 9. Download cap arrays for CPU push_relabel ───────────────────────
        h_cap_src   = self.d_cap_src.copy_to_host()
        h_cap_snk   = self.d_cap_snk.copy_to_host()
        h_cap_right = self.d_cap_right.copy_to_host()
        h_cap_down  = self.d_cap_down.copy_to_host()

        # ── 10. Push-Relabel max-flow (CPU parallel @njit) ────────────────────
        labeling = push_relabel(
            h_cap_src, h_cap_snk, h_cap_right, h_cap_down,
            H, W, PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ,
        )

        fg = (labeling == 0).reshape(int(H), int(W)).astype(np.uint8)
        np.multiply(fg, np.uint8(255), out=self._final_mask)

        # ── 11. Morphological cleanup (CPU @njit) ─────────────────────────────
        morphological_close(self._final_mask, self._morph_tmp1, self._morph_tmp2,
                            H, W, radius=3)
        morphological_open(self._morph_tmp2, self._morph_tmp1, self._final_mask,
                           H, W, radius=2)

        np.copyto(self._final_mask,
                  largest_component(self._final_mask, H, W))

        # ── 12. Blurred composite (GaussianBlur CPU, compose GPU) ─────────────
        h_frame   = d_frame.copy_to_host()
        ks        = self.blur_ks
        h_blurred = cv2.GaussianBlur(h_frame, (ks, ks), 0)
        self.d_blurred.copy_to_device(h_blurred)
        self.d_final_mask.copy_to_device(self._final_mask)
        _compose_blur_cuda[self._grid, self._BLOCK](
            d_frame, self.d_blurred, self.d_final_mask, self.d_composite, H, W)

    def _step_device_profiler(self, d_frame, d_bg_prob, profiling):
        """Full GrabCut pipeline.  Both inputs must be device arrays.

        d_frame   : (H, W, 3) float32  device
        d_bg_prob : (H, W)    float32  device
        Returns   : (final_mask, composite) as host uint8 arrays.
        """
        H, W = np.int32(self.H), np.int32(self.W)

        profiling_d = dict()
        profiler_func = line_measurer_cuda if profiling else (lambda func, *args, **kwargs: func(*args, **kwargs))

        # ── 1. GC label map ───────────────────────────────────────────────────

        _, profiling_d['make gc'] = line_measurer_cuda(_make_gc_mask_cuda[self._grid, self._BLOCK],
            d_bg_prob, self.d_gc_mask, H, W)

        # ── 2. GMM fit (E + M-step, GPU) ──────────────────────────────────────
        profiling_d['bg fit'] = self._bg_gmm.fit(d_frame, self.d_gc_mask, is_fg=False)
        profiling_d['fg fit'] = self._fg_gmm.fit(d_frame, self.d_gc_mask, is_fg=True)
        
        # ── 3. Neg-log-prob maps (GPU) ────────────────────────────────────────
        profiling_d['nlp bg'] = self._bg_gmm.neg_log_prob(d_frame, self.d_nlp_bg)
        profiling_d['nlp fg'] = self._fg_gmm.neg_log_prob(d_frame, self.d_nlp_fg)

        # ── 4. Beta scalar (GPU reduction → 1 scalar download) ────────────────
        self.d_beta_acc.copy_to_device(self._h_beta_zero)

        _, profiling_d['beta accumulate'] = line_measurer_cuda(_calc_beta_acc_cuda[self._grid, self._BLOCK],
            d_frame, self.d_beta_acc, H, W)
        # cuda.synchronize()

        beta = self.d_beta_acc.copy_to_host()[0]

        # ── 5. N-weights (GPU) ────────────────────────────────────────────────
        _, profiling_d['calc nweights'] = line_measurer_cuda(_calc_nweights_cuda[self._grid, self._BLOCK],
            d_frame, beta, np.float32(self.gamma),
            self.d_leftW, self.d_upleftW, self.d_upW, self.d_uprightW, H, W)
        # cuda.synchronize()

        # ── 6. lam scalar (download leftW + upW for max) ──────────────────────
        h_leftW = self.d_leftW.copy_to_host()
        h_upW   = self.d_upW.copy_to_host()
        max_nw  = max(float(h_leftW.max()), float(h_upW.max()))
        lam     = np.float32(max_nw * LAM_FACTOR
                             if max_nw > 0.0
                             else float(self.gamma) * LAM_FACTOR)

        # ── 7. Build t-links (GPU) ────────────────────────────────────────────
        _, profiling_d['build tlinks'] = line_measurer_cuda(_build_tlinks_cuda[self._grid, self._BLOCK],
            self.d_gc_mask, self.d_nlp_bg, self.d_nlp_fg, lam,
            self.d_cap_src, self.d_cap_snk, H, W)

        # ── 8. Build n-links (GPU) ────────────────────────────────────────────
        _, profiling_d['build nlinks'] = line_measurer_cuda(_build_nlinks_cuda[self._grid, self._BLOCK],
            self.d_leftW, self.d_upW,
            self.d_cap_right, self.d_cap_down, H, W)
        # cuda.synchronize()

        # ── 9. Download cap arrays for CPU push_relabel ───────────────────────
        h_cap_src   = self.d_cap_src.copy_to_host()
        h_cap_snk   = self.d_cap_snk.copy_to_host()
        h_cap_right = self.d_cap_right.copy_to_host()
        h_cap_down  = self.d_cap_down.copy_to_host()

        # ── 10. Push-Relabel max-flow (CPU parallel @njit) ────────────────────
        labeling, profiling_d['push relabel'] = line_measurer(push_relabel,
            h_cap_src, h_cap_snk, h_cap_right, h_cap_down,
            H, W, PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ,
        )

        print("GR ", profiling_d['push relabel'])
        # print('\n' + '---'*24)
        # for k, v in profiling_d.items():
        #     print(f'{k}\t{v}')

        fg = (labeling == 0).reshape(int(H), int(W)).astype(np.uint8)
        np.multiply(fg, np.uint8(255), out=self._final_mask)

        # ── 11. Morphological cleanup (CPU @njit) ─────────────────────────────
        morphological_close(self._final_mask, self._morph_tmp1, self._morph_tmp2,
                            H, W, radius=3)
        morphological_open(self._morph_tmp2, self._morph_tmp1, self._final_mask,
                           H, W, radius=2)

        np.copyto(self._final_mask,
                  largest_component(self._final_mask, H, W))

        # ── 12. Blurred composite (GaussianBlur CPU, compose GPU) ─────────────
        h_frame   = d_frame.copy_to_host()
        ks        = self.blur_ks
        h_blurred = cv2.GaussianBlur(h_frame, (ks, ks), 0)
        self.d_blurred.copy_to_device(h_blurred)
        self.d_final_mask.copy_to_device(self._final_mask)
        _compose_blur_cuda[self._grid, self._BLOCK](
            d_frame, self.d_blurred, self.d_final_mask, self.d_composite, H, W)

# ── CUDA device helpers ───────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _expf(x):
    return cuda.libdevice.expf(x)


# ── CUDA kernels ──────────────────────────────────────────────────────────────

@cuda.jit
def _make_gc_mask_cuda(bg_prob, gc_mask, H, W):
    """Classify each pixel as GC_PR_BGD (2) or GC_PR_FGD (3) from bg_prob."""
    idx = cuda.grid(1)
    if idx >= H * W:
        return
    y = idx // W
    x = idx % W
    if bg_prob[y, x] >= np.float32(0.70):
        gc_mask[y, x] = np.uint8(2)   # GC_PR_BGD
    else:
        gc_mask[y, x] = np.uint8(3)   # GC_PR_FGD


@cuda.jit
def _calc_beta_acc_cuda(frame, beta_acc, H, W):
    """Accumulate sum of squared 8-neighbour colour differences into beta_acc[0]."""
    idx = cuda.grid(1)
    if idx >= H * W:
        return
    y = idx // W
    x = idx % W

    b0 = np.float32(frame[y, x, 0])
    g0 = np.float32(frame[y, x, 1])
    r0 = np.float32(frame[y, x, 2])
    local = np.float32(0.0)

    if x > 0:
        db = b0 - np.float32(frame[y, x - 1, 0])
        dg = g0 - np.float32(frame[y, x - 1, 1])
        dr = r0 - np.float32(frame[y, x - 1, 2])
        local += db*db + dg*dg + dr*dr
    if y > 0 and x > 0:
        db = b0 - np.float32(frame[y - 1, x - 1, 0])
        dg = g0 - np.float32(frame[y - 1, x - 1, 1])
        dr = r0 - np.float32(frame[y - 1, x - 1, 2])
        local += db*db + dg*dg + dr*dr
    if y > 0:
        db = b0 - np.float32(frame[y - 1, x, 0])
        dg = g0 - np.float32(frame[y - 1, x, 1])
        dr = r0 - np.float32(frame[y - 1, x, 2])
        local += db*db + dg*dg + dr*dr
    if y > 0 and x + 1 < W:
        db = b0 - np.float32(frame[y - 1, x + 1, 0])
        dg = g0 - np.float32(frame[y - 1, x + 1, 1])
        dr = r0 - np.float32(frame[y - 1, x + 1, 2])
        local += db*db + dg*dg + dr*dr

    cuda.atomic.add(beta_acc, 0, local)
    denom    = 4 * int(W) * int(H) - 3 * int(W) - 3 * int(H) + 2

    beta_acc[0] = np.float32(0.0 if beta_acc[0] <= 1e-12
                              else 1.0 / (2.0 * beta_acc[0] / denom))
    
@cuda.jit
def _calc_nweights_cuda(frame, beta, gamma, leftW, upleftW, upW, uprightW, H, W):
    """Per-pixel 4-directional neighbour weights: gamma·expf(-beta·||diff||²)."""
    idx = cuda.grid(1)
    if idx >= H * W:
        return
    y = idx // W
    x = idx % W

    gds2 = gamma / np.float32(1.4142135623730951)
    b0   = frame[y, x, 0]
    g0   = frame[y, x, 1]
    r0   = frame[y, x, 2]

    if x > 0:
        db = b0 - frame[y, x - 1, 0]
        dg = g0 - frame[y, x - 1, 1]
        dr = r0 - frame[y, x - 1, 2]
        leftW[y, x] = gamma * _expf(-beta * (db*db + dg*dg + dr*dr))
    else:
        leftW[y, x] = np.float32(0.0)

    if x > 0 and y > 0:
        db = b0 - frame[y - 1, x - 1, 0]
        dg = g0 - frame[y - 1, x - 1, 1]
        dr = r0 - frame[y - 1, x - 1, 2]
        upleftW[y, x] = gds2 * _expf(-beta * (db*db + dg*dg + dr*dr))
    else:
        upleftW[y, x] = np.float32(0.0)

    if y > 0:
        db = b0 - frame[y - 1, x, 0]
        dg = g0 - frame[y - 1, x, 1]
        dr = r0 - frame[y - 1, x, 2]
        upW[y, x] = gamma * _expf(-beta * (db*db + dg*dg + dr*dr))
    else:
        upW[y, x] = np.float32(0.0)

    if x + 1 < W and y > 0:
        db = b0 - frame[y - 1, x + 1, 0]
        dg = g0 - frame[y - 1, x + 1, 1]
        dr = r0 - frame[y - 1, x + 1, 2]
        uprightW[y, x] = gds2 * _expf(-beta * (db*db + dg*dg + dr*dr))
    else:
        uprightW[y, x] = np.float32(0.0)


@cuda.jit
def _build_tlinks_cuda(gc_mask, nlp_bg, nlp_fg, lam, cap_src, cap_snk, H, W):
    """Terminal-link capacities — exact grabcut.cpp semantics."""
    idx = cuda.grid(1)
    if idx >= H * W:
        return
    y = idx // W
    x = idx % W
    m = gc_mask[y, x]
    if m == np.uint8(0):           # GC_BGD
        cap_src[idx] = np.float32(0.0)
        cap_snk[idx] = lam
    elif m == np.uint8(1):         # GC_FGD
        cap_src[idx] = lam
        cap_snk[idx] = np.float32(0.0)
    else:                          # GC_PR_BGD / GC_PR_FGD
        cap_src[idx] = nlp_bg[y, x]
        cap_snk[idx] = nlp_fg[y, x]


@cuda.jit
def _build_nlinks_cuda(leftW, upW, cap_right, cap_down, H, W):
    """Map 2-D weight arrays into flat capacity arrays for push_relabel."""
    idx = cuda.grid(1)
    if idx >= H * W:
        return
    y = idx // W
    x = idx % W
    if x + 1 < W:
        cap_right[idx] = leftW[y, x + 1]
    else:
        cap_right[idx] = np.float32(0.0)
    if y + 1 < H:
        cap_down[idx] = upW[y + 1, x]
    else:
        cap_down[idx] = np.float32(0.0)


@cuda.jit
def _compose_blur_cuda(frame, blurred, fg_mask, out, H, W):
    """Composite: fg pixels from frame (float32→uint8), bg from blurred."""
    idx = cuda.grid(1)
    if idx >= H * W:
        return
    y = idx // W
    x = idx % W
    if fg_mask[y, x] > np.uint8(0):
        out[y, x, 0] = np.uint8(frame[y, x, 0])
        out[y, x, 1] = np.uint8(frame[y, x, 1])
        out[y, x, 2] = np.uint8(frame[y, x, 2])
    else:
        out[y, x, 0] = np.uint8(blurred[y, x, 0])
        out[y, x, 1] = np.uint8(blurred[y, x, 1])
        out[y, x, 2] = np.uint8(blurred[y, x, 2])


def warmup_grabcut_jit():
    """Pre-compile all CUDA kernels and CPU JIT functions on tiny dummy data."""
    warmup_push_relabel()
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
    d_beta_acc = cuda.to_device(np.zeros(1, np.float32))

    _make_gc_mask_cuda   [grid, BLOCK](d_bg_prob, d_gc_mask, H, W)
    _calc_beta_acc_cuda  [grid, BLOCK](d_frame, d_beta_acc, H, W)
    _calc_nweights_cuda  [grid, BLOCK](d_frame, np.float32(0.1), np.float32(50.0),
                                       d_leftW, d_upleftW, d_upW, d_uprightW, H, W)
    _build_tlinks_cuda   [grid, BLOCK](d_gc_mask, d_nlp_bg, d_nlp_fg,
                                       np.float32(1.0), d_cap_src, d_cap_snk, H, W)
    _build_nlinks_cuda   [grid, BLOCK](d_leftW, d_upW, d_cap_right, d_cap_down, H, W)
    _compose_blur_cuda   [grid, BLOCK](d_frame, d_blurred, d_final, d_out, H, W)
    cuda.synchronize()
