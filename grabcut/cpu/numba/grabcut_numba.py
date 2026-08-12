import numpy as np
import cv2
from numba import njit, prange

from grabcut.grabcut_common import GrabCut_Base
from .push_relabel import push_relabel, warmup_push_relabel
from .morphology import *
from settings import *

from utils import line_measurer



class GrabCut_Numba(GrabCut_Base):
    """MOG2 + dual Rother-2004 GMM + Numba parallel Push-Relabel.

    Parameters
    ----------
    gmm         : GMM_CPU_NUMBA — background subtractor.
    roi_rect    : (x, y, w, h) seed ROI.  ``None`` → 60 %×70 % centred.
    gamma       : smoothness weight (default 50).
    blur_ksize  : GaussianBlur kernel size for background composite.
    dynamic_roi : bool — derive ROI each frame from MOG2 bg_prob.
    **roi_kwargs: forwarded to ``DynamicROI`` (fg_threshold, margin,
                  smoothing, min_fg_frac, clean_kernel).
    """

    def apply(self, frame: np.ndarray, bg_prob: np.ndarray, profiling=False):
        """Run one full pipeline frame.

        Parameters
        ----------
        frame_bgr    : (H, W, 3) uint8
        update_alpha : MOG2 learning rate override (-1 = auto ramp)

        Returns
        -------
        mog2_mask   : (H, W) uint8
        bg_prob     : (H, W) float32
        final_mask  : (H, W) uint8   (255 = foreground)
        composite   : (H, W, 3) uint8
        elapsed_mog2: float (seconds)
        """
        H, W = self.H, self.W

        profiling_d = dict()
        profiler_func = line_measurer if profiling else (lambda func, *args, **kwargs: func(*args, **kwargs))

        # 
        _, profiling_d['make_gc'] = profiler_func(make_gc_mask, bg_prob, self._gc_mask)
 
        # 4 ── Dual spatial GMMs
        img_f64 = frame.astype(np.float32)
        _, profiling_d['bg fit'] = profiler_func(self._bg_gmm.fit, img_f64, self._gc_mask, is_fg=False)
        _, profiling_d['fg fit'] = profiler_func(self._fg_gmm.fit, img_f64, self._gc_mask, is_fg=True)

        # 5 ── Neg-log-likelihood maps
        _, profiling_d['bg nlp'] = profiler_func(self._bg_gmm.neg_log_prob, img_f64, self._nlp_bg)
        _, profiling_d['fg nlp'] = profiler_func(self._fg_gmm.neg_log_prob, img_f64, self._nlp_fg)

        # 6 ── Beta + n-weights
        beta, profiling_d['calc beta'] = profiler_func(calc_beta, img_f64)
        _, profiling_d['calc nweights'] = profiler_func(calc_nweights, img_f64, beta, self.gamma,
                      self._leftW, self._upleftW, self._upW, self._uprightW)

        max_nw, profiling_d['max nw'] = profiler_func(lambda: max(float(self._leftW.max()), float(self._upW.max())))
        lam    = (max_nw * LAM_FACTOR
                  if max_nw > 0.0 else float(self.gamma) * float(LAM_FACTOR))

        # 7 ── Build graph capacities
        _, profiling_d['build tlinks'] = profiler_func(build_tlinks, self._gc_mask, self._nlp_bg, self._nlp_fg, lam,
                     self._cap_src, self._cap_snk)
        
        _, profiling_d['build nlinks'] = profiler_func(build_nlinks, self._leftW.astype(np.float32),
                     self._upW.astype(np.float32),
                     np.int32(H), np.int32(W),
                     self._cap_right, self._cap_down)

        # 8 ── Push-Relabel max-flow
        labeling, profiling_d['push relabel'] = profiler_func(push_relabel, 
            self._cap_src, self._cap_snk,
            self._cap_right, self._cap_down,
            np.int32(H), np.int32(W),
            PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ,
        )
        fg = (labeling == 0).reshape(H, W).astype(np.uint8)
        np.multiply(fg, np.uint8(255), out=self._final_mask)

        # 9 ── Morphological cleanup
        _, profiling_d['morph close'] = profiler_func(morphological_close, self._final_mask, self._morph_tmp1, self._morph_tmp2,
                            np.int32(H), np.int32(W), radius=3)
        _, profiling_d['morph open'] = profiler_func(morphological_open, self._morph_tmp2, self._morph_tmp1, self._final_mask,
                           np.int32(H), np.int32(W), radius=2)

        lrg_comp, profiling_d['large component'] = profiler_func(largest_component, self._final_mask, np.int32(H), np.int32(W))

        np.copyto(self._final_mask, lrg_comp)

        # 10 ── Blurred composite
        ks = self.blur_ks
        self._blurred[:] = cv2.GaussianBlur(frame, (ks, ks), 0)
        compose_blur(frame, self._blurred, self._final_mask, self._composite)

        return self._final_mask, self._composite, profiling_d

@njit(parallel=True, cache=True)
def make_gc_mask(bg_prob, gc_mask,
                 bg_hard_thresh=np.float32(0.70),
                 fg_hard_thresh=np.float32(0.20)):
    """Derive GC label map (0/2/3) from MOG2 bg_prob."""
    H = bg_prob.shape[0]
    W = bg_prob.shape[1]
    for y in prange(H):
        for x in range(W):
            if bg_prob[y, x] >= bg_hard_thresh:
                gc_mask[y, x] = np.uint8(2)          # GC_PR_BGD
            else:
                gc_mask[y, x] = np.uint8(3)          # GC_PR_FGD

@njit(parallel=True, cache=True)
def calc_beta(img):
    """Mean squared colour distance over 8-connected pairs → beta scalar."""
    h, w, _ = img.shape
    beta = np.float64(0.0)
    for y in prange(h):
        for x in range(w):
            c0 = img[y, x]
            if x > 0:
                d = c0 - img[y, x - 1]
                beta += d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
            if y > 0 and x > 0:
                d = c0 - img[y - 1, x - 1]
                beta += d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
            if y > 0:
                d = c0 - img[y - 1, x]
                beta += d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
            if y > 0 and x < w - 1:
                d = c0 - img[y - 1, x + 1]
                beta += d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
    denom = 4 * w * h - 3 * w - 3 * h + 2
    if beta <= 1e-12:
        return np.float64(0.0)
    return np.float64(1.0 / (2.0 * (beta / denom)))

@njit(parallel=True, cache=True)
def calc_nweights(img, beta, gamma, leftW, upleftW, upW, uprightW):
    """Per-pixel 4-directional neighbour weights: gamma·exp(-beta·||diff||²)."""
    h, w, _ = img.shape
    gds2 = gamma / np.float64(1.4142135623730951)
    for y in prange(h):
        for x in range(w):
            c = img[y, x]
            if x > 0:
                d = c - img[y, x - 1]
                leftW[y, x] = gamma * np.exp(
                    -beta * (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]))
            else:
                leftW[y, x] = np.float64(0.0)
            if x > 0 and y > 0:
                d = c - img[y - 1, x - 1]
                upleftW[y, x] = gds2 * np.exp(
                    -beta * (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]))
            else:
                upleftW[y, x] = np.float64(0.0)
            if y > 0:
                d = c - img[y - 1, x]
                upW[y, x] = gamma * np.exp(
                    -beta * (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]))
            else:
                upW[y, x] = np.float64(0.0)
            if x + 1 < w and y > 0:
                d = c - img[y - 1, x + 1]
                uprightW[y, x] = gds2 * np.exp(
                    -beta * (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]))
            else:
                uprightW[y, x] = np.float64(0.0)

@njit(parallel=True, cache=True)
def compose_blur(frame, blurred, fg_mask, out):
    """Composite: fg pixels from ``frame``, bg pixels from ``blurred``."""
    H = frame.shape[0]
    W = frame.shape[1]
    for y in prange(H):
        for x in range(W):
            if fg_mask[y, x] > np.uint8(0):
                out[y, x, 0] = frame[y, x, 0]
                out[y, x, 1] = frame[y, x, 1]
                out[y, x, 2] = frame[y, x, 2]
            else:
                out[y, x, 0] = blurred[y, x, 0]
                out[y, x, 1] = blurred[y, x, 1]
                out[y, x, 2] = blurred[y, x, 2]

@njit(parallel=True, cache=True)
def build_tlinks(gc_mask, nlp_bg, nlp_fg, lam, cap_src, cap_snk):
    """Fill terminal-link capacities — exact grabcut.cpp semantics."""
    H = gc_mask.shape[0]
    W = gc_mask.shape[1]
    lam_f32 = np.float32(lam)
    for y in prange(H):
        for x in range(W):
            n = y * W + x
            m = gc_mask[y, x]
            if m == np.uint8(0):           # GC_BGD
                cap_src[n] = np.float32(0.0)
                cap_snk[n] = lam_f32
            elif m == np.uint8(1):         # GC_FGD
                cap_src[n] = lam_f32
                cap_snk[n] = np.float32(0.0)
            else:                          # GC_PR_BGD / GC_PR_FGD
                cap_src[n] = nlp_bg[y, x]
                cap_snk[n] = nlp_fg[y, x]

@njit(parallel=True, cache=True)
def build_nlinks(leftW_f32, upW_f32, H, W, cap_right, cap_down):
    """Map 2-D weight arrays into flat capacity arrays for Push-Relabel."""
    for y in prange(H):
        for x in range(W):
            n = y * W + x
            cap_right[n] = leftW_f32[y, x + 1] if x + 1 < W else np.float32(0.0)
            cap_down[n]  = upW_f32[y + 1, x]   if y + 1 < H else np.float32(0.0)

def warmup_grabcut_jit():
        warmup_push_relabel()
        warmup_morph()
