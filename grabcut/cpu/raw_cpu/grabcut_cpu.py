import numpy as np
import cv2

from grabcut.grabcut_common import GrabCut_Base
from .push_relabel import push_relabel
from .morphology import *
from settings import *

class GrabCut_CPU(GrabCut_Base):
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

        # 
        make_gc_mask(bg_prob, self._gc_mask)
    
        # 4 ── Dual spatial GMMs
        img_f64 = frame.astype(np.float32)
        self._bg_gmm.fit(img_f64, self._gc_mask, is_fg=False)
        self._fg_gmm.fit(img_f64, self._gc_mask, is_fg=True)

        # 5 ── Neg-log-likelihood maps
        self._bg_gmm.neg_log_prob(img_f64, self._nlp_bg)
        self._fg_gmm.neg_log_prob(img_f64, self._nlp_fg)

        # 6 ── Beta + n-weights
        beta = calc_beta(img_f64)
        calc_nweights(img_f64, beta, self.gamma,
                        self._leftW, self._upleftW, self._upW, self._uprightW)

        max_nw = max(float(self._leftW.max()), float(self._upW.max()))
        lam    = (max_nw * LAM_FACTOR
                    if max_nw > 0.0 else float(self.gamma) * float(LAM_FACTOR))

        # 7 ── Build graph capacities
        build_tlinks(self._gc_mask, self._nlp_bg, self._nlp_fg, lam,
                        self._cap_src, self._cap_snk)
        
        build_nlinks(self._leftW.astype(np.float32),
                        self._upW.astype(np.float32),
                        np.int32(H), np.int32(W),
                        self._cap_right, self._cap_down)

        # 8 ── Push-Relabel max-flow
        labeling = push_relabel(
            self._cap_src, self._cap_snk,
            self._cap_right, self._cap_down,
            np.int32(H), np.int32(W),
            PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ,
        )
        fg = (labeling == 0).reshape(H, W).astype(np.uint8)
        np.multiply(fg, np.uint8(255), out=self._final_mask)

        # 9 ── Morphological cleanup
        morphological_close(self._final_mask, self._morph_tmp1, self._morph_tmp2,
                            np.int32(H), np.int32(W), radius=3)
        morphological_open(self._morph_tmp2, self._morph_tmp1, self._final_mask,
                            np.int32(H), np.int32(W), radius=2)
        np.copyto(self._final_mask, largest_component(self._final_mask,
                                                        np.int32(H), np.int32(W)))

        # 10 ── Blurred composite
        ks = self.blur_ks
        self._blurred[:] = cv2.GaussianBlur(frame, (ks, ks), 0)
        compose_blur(frame, self._blurred, self._final_mask, self._composite)

        return self._final_mask, self._composite

def make_gc_mask(bg_prob, gc_mask,
                    bg_hard_thresh=np.float32(0.70),
                    fg_hard_thresh=np.float32(0.20)):
    """Derive GC label map (0/2/3) from MOG2 bg_prob."""
    H = bg_prob.shape[0]
    W = bg_prob.shape[1]
    for y in range(H):
        for x in range(W):
            if bg_prob[y, x] >= bg_hard_thresh:
                gc_mask[y, x] = np.uint8(2)          # GC_PR_BGD
            else:
                gc_mask[y, x] = np.uint8(3)          # GC_PR_FGD

def calc_beta(img):
    """Mean squared colour distance over 8-connected pairs → beta scalar."""
    h, w, _ = img.shape
    beta = np.float64(0.0)
    for y in range(h):
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

def calc_nweights(img, beta, gamma, leftW, upleftW, upW, uprightW):
    """Per-pixel 4-directional neighbour weights: gamma·exp(-beta·||diff||²)."""
    h, w, _ = img.shape
    gds2 = gamma / np.float64(1.4142135623730951)
    for y in range(h):
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

def compose_blur(frame, blurred, fg_mask, out):
    """Composite: fg pixels from ``frame``, bg pixels from ``blurred``."""
    H = frame.shape[0]
    W = frame.shape[1]
    for y in range(H):
        for x in range(W):
            if fg_mask[y, x] > np.uint8(0):
                out[y, x, 0] = frame[y, x, 0]
                out[y, x, 1] = frame[y, x, 1]
                out[y, x, 2] = frame[y, x, 2]
            else:
                out[y, x, 0] = blurred[y, x, 0]
                out[y, x, 1] = blurred[y, x, 1]
                out[y, x, 2] = blurred[y, x, 2]

def build_tlinks(gc_mask, nlp_bg, nlp_fg, lam, cap_src, cap_snk):
    """Fill terminal-link capacities — exact grabcut.cpp semantics."""
    H = gc_mask.shape[0]
    W = gc_mask.shape[1]
    lam_f32 = np.float32(lam)
    for y in range(H):
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

def build_nlinks(leftW_f32, upW_f32, H, W, cap_right, cap_down):
        """Map 2-D weight arrays into flat capacity arrays for Push-Relabel."""
        for y in range(H):
            for x in range(W):
                n = y * W + x
                cap_right[n] = leftW_f32[y, x + 1] if x + 1 < W else np.float32(0.0)
                cap_down[n]  = upW_f32[y + 1, x]   if y + 1 < H else np.float32(0.0)