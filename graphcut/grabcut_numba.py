"""GrabCut pipeline: MOG2 background model + Rother-2004 dual-GMM → Push-Relabel.

Pipeline (one frame)
--------------------
1. MOG2 step  → binary mask (0=bg, 255=fg) + bg_prob (H,W)
2. Derive GC label map from MOG2 mask + ROI:
       outside ROI           → GC_BGD  (0) — hard background
       inside ROI, bg_prob≥t → GC_PR_BGD (2) — probably background
       inside ROI, bg_prob<t → GC_PR_FGD (3) — probably foreground
3. Fit background GMM from GC_BGD + GC_PR_BGD pixels (Rother 2004 5-component).
4. Fit foreground GMM from GC_PR_FGD pixels.
   (Background colour model = MOG2 gmm is NOT touched — clean separation.)
5. Build t-links exactly as grabcut.cpp:
       GC_BGD    → fromSource=0, toSink=lambda  (hard background)
       GC_FGD    → fromSource=lambda, toSink=0  (hard foreground — unused here)
       uncertain → fromSource=-log(bgdGMM(color)), toSink=-log(fgdGMM(color))
6. Build n-links (4-connected, gamma*exp(-beta*||diff||^2)).
7. Parallel Push-Relabel → labeling (SOURCE-side = foreground, grabcut.cpp semantics).
8. Morphological close→open + largest_component cleanup.
9. Compose: foreground pixels from original frame, background pixels blurred.

Reference
---------
grabcut.cpp (OpenCV), constructGCGraph(), assignGMMsComponents(), learnGMMs().
"""
import warnings

import numpy as np
from numba import njit, prange

from graphcut.push_relabel_numba import push_relabel, warmup_push_relabel
from graphcut.morph_numba import (morphological_close, morphological_open,
                         largest_component, warmup_morph)
from graphcut.fgd_gmm_numba import GrabCutGMM, warmup_fgd_gmm
from gmm.mog2_common import to_planar

GAMMA = np.float64(50.0)
# lambda = 9 * gamma  (Rother 2004, Eq. 7)
LAM_FACTOR = np.float64(9.0)

# Convergence takes roughly one outer iteration per pixel of image diameter:
# measured 119 at 60x80, 783 at 240x320, 1027 at 480x640. A fixed 200 was
# therefore fine on the tiny test grids and silently returned a 0%-correct
# labelling at 240x320 and above, which is the size this actually runs at.
# Scale with (H + W) and keep a wide margin; the loop exits on convergence, so
# a generous cap costs nothing when it is not needed.
PUSH_RELABEL_ITER_PER_DIAMETER = 8
PUSH_RELABEL_MIN_ITER          = 1000
PUSH_RELABEL_RELABEL_FREQ      = 20

# MOG2 bg_prob threshold for deriving the initial GC label map
BG_HARD_THRESH  = np.float32(0.70)   # above → GC_PR_BGD
FG_HARD_THRESH  = np.float32(0.20)   # below → GC_PR_FGD


# ── Beta & n-weight kernels (unchanged from OpenCV reference) ────────────────

@njit(parallel=True, cache=True)
def calc_beta_numba(img):
    """Mean squared colour distance over 8-connected pairs → beta scalar."""
    h, w, c = img.shape
    beta = 0.0
    for y in prange(h):
        for x in range(w):
            color0 = img[y, x]
            if x > 0:
                diff = color0 - img[y, x - 1]
                beta += diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
            if y > 0 and x > 0:
                diff = color0 - img[y - 1, x - 1]
                beta += diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
            if y > 0:
                diff = color0 - img[y - 1, x]
                beta += diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
            if y > 0 and x < w - 1:
                diff = color0 - img[y - 1, x + 1]
                beta += diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
    denom = 4 * w * h - 3 * w - 3 * h + 2
    if beta <= 1e-12:
        return 0.0
    return 1.0 / (2.0 * (beta / denom))


@njit(parallel=True, cache=True)
def calc_nweights_numba(img, beta, gamma, leftW, upleftW, upW, uprightW):
    """Per-pixel 4-directional neighbour weights: gamma*exp(-beta*||diff||²)."""
    h, w, c = img.shape
    gds2 = gamma / 1.4142135623730951
    for y in prange(h):
        for x in range(w):
            color = img[y, x]
            if x > 0:
                diff = color - img[y, x - 1]
                leftW[y, x] = gamma * np.exp(-beta * (diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
            else:
                leftW[y, x] = 0.0
            if x > 0 and y > 0:
                diff = color - img[y - 1, x - 1]
                upleftW[y, x] = gds2 * np.exp(-beta * (diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
            else:
                upleftW[y, x] = 0.0
            if y > 0:
                diff = color - img[y - 1, x]
                upW[y, x] = gamma * np.exp(-beta * (diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
            else:
                upW[y, x] = 0.0
            if x + 1 < w and y > 0:
                diff = color - img[y - 1, x + 1]
                uprightW[y, x] = gds2 * np.exp(-beta * (diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
            else:
                uprightW[y, x] = 0.0


# ── GC label map: derived from MOG2 bg_prob + ROI ───────────────────────────

@njit(parallel=True, cache=True)
def _make_gc_mask(bg_prob, roi_mask, gc_mask,
                  bg_hard_thresh=np.float32(0.70),
                  fg_hard_thresh=np.float32(0.20)):
    """Derive GC label map (0/2/3) from MOG2 bg_prob and ROI mask.

    GC_BGD   = 0 : definite background (outside ROI)
    GC_FGD   = 1 : definite foreground (unused — no user scribbles)
    GC_PR_BGD= 2 : probable background (inside ROI, high bg_prob)
    GC_PR_FGD= 3 : probable foreground (inside ROI, low bg_prob)
    """
    H = bg_prob.shape[0]
    W = bg_prob.shape[1]
    for y in prange(H):
        for x in range(W):
            if roi_mask[y, x] == np.uint8(0):
                gc_mask[y, x] = np.uint8(0)   # GC_BGD
            elif bg_prob[y, x] >= bg_hard_thresh:
                gc_mask[y, x] = np.uint8(2)   # GC_PR_BGD
            else:
                gc_mask[y, x] = np.uint8(3)   # GC_PR_FGD


# ── T-link builder: exact grabcut.cpp constructGCGraph semantics ─────────────

@njit(parallel=True, cache=True)
def _build_tlinks(gc_mask, nlp_bg, nlp_fg, lam, cap_src, cap_snk):
    """Fill terminal-link capacities from dual-GMM neg-log-likelihoods.

    Matches grabcut.cpp constructGCGraph exactly:
        GC_BGD    : fromSource=0,       toSink=lam    (pixel is hard background)
        GC_FGD    : fromSource=lam,     toSink=0      (pixel is hard foreground)
        GC_PR_*   : fromSource=-log(bgGMM), toSink=-log(fgGMM)

    In grabcut.cpp: SOURCE side = foreground after min-cut.
    Our Push-Relabel returns 1 for pixels reachable from SINK → background side.
    We flip: cap_src = toSink, cap_snk = fromSource
    so that SOURCE-side pixels (1 after min-cut) map to foreground in our code.

    Actually keep the exact grabcut.cpp polarity:
        cap_src (SOURCE→pixel) is cut when pixel goes to SINK (background).
        cap_src = fromSource = cost of calling the pixel foreground.
        cap_snk = toSink    = cost of calling the pixel background.

    grabcut.cpp: fromSource = -log(bgGMM), toSink = -log(fgGMM).
    After graph cut: inSourceSegment → foreground.
    Our PR: height_label < INF → reachable from SINK → SINK-side → background.
    So we set:
        cap_src[n] = -log(bgGMM)   cost to cut SOURCE→n (calling n foreground)
        cap_snk[n] = -log(fgGMM)   cost to cut n→SINK   (calling n background)
    Then SINK-side (our labeling=1) = background, SOURCE-side = foreground.
    We invert labeling at the end: fg_mask = (labeling == 0).
    """
    H = gc_mask.shape[0]
    W = gc_mask.shape[1]
    lam_f32 = np.float32(lam)

    for y in prange(H):
        for x in range(W):
            n = y * W + x
            m = gc_mask[y, x]
            if m == np.uint8(0):          # GC_BGD — hard background
                cap_src[n] = np.float32(0.0)
                cap_snk[n] = lam_f32
            elif m == np.uint8(1):        # GC_FGD — hard foreground
                cap_src[n] = lam_f32
                cap_snk[n] = np.float32(0.0)
            else:                         # GC_PR_BGD or GC_PR_FGD — soft
                # fromSource = -log(bgGMM(color))
                # toSink     = -log(fgGMM(color))
                #
                # Clamped at zero: the GMM density is unnormalised —
                # coef/sqrt(det) * exp(...) with no (2*pi)^(3/2) — so a tight
                # component (det driven to the 1e-6 singular floor) returns a
                # value above 1 and -log of it is negative. Max-flow is only
                # defined for non-negative capacities; a negative one makes
                # push-relabel return a cut that is not minimal, silently.
                src = nlp_bg[y, x]
                snk = nlp_fg[y, x]
                cap_src[n] = src if src > np.float32(0.0) else np.float32(0.0)
                cap_snk[n] = snk if snk > np.float32(0.0) else np.float32(0.0)


# ── N-link builder ───────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _build_nlinks(leftW_f32, upW_f32, H, W, cap_right, cap_down):
    """Map 2-D weight arrays into flat capacity arrays for Push-Relabel."""
    for y in prange(H):
        for x in range(W):
            n = y * W + x
            cap_right[n] = leftW_f32[y, x + 1] if x + 1 < W else np.float32(0.0)
            cap_down[n]  = upW_f32[y + 1, x]   if y + 1 < H else np.float32(0.0)


# ── Background composition kernel ────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _compose_blur(frame, blurred, fg_mask, out):
    """Composite: fg pixels from `frame`, bg pixels from `blurred`.

    All arrays (H, W, 3) uint8.  Written into `out` in-place.
    """
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


# ── Main pipeline class ──────────────────────────────────────────────────────

class GrabCutPipeline:
    """MOG2 background model + dual Rother-2004 GMM → Push-Relabel segmentation.

    Background colour model  : MOG2 (temporal, gives bg_prob per pixel).
    Background spatial GMM   : GrabCutGMM fit each frame from GC_PR_BGD pixels.
    Foreground spatial GMM   : GrabCutGMM fit each frame from GC_PR_FGD pixels.

    The two GrabCutGMMs operate on per-frame colour appearance; the MOG2 model
    provides the temporal prior that initialises the GC label map.

    Parameters
    ----------
    gmm       : GMM_CPU_NUMBA — background subtractor (must not be shared).
    roi_rect  : (x, y, w, h) — region of interest; outside = hard background.
    gamma     : smoothness weight (default 50, matches OpenCV grabCut).
    blur_ksize: GaussianBlur kernel size for the background composite.
    """

    def __init__(self, gmm, roi_rect, gamma=GAMMA, blur_ksize=15):
        if not getattr(gmm, "FILLS_BG_PROB", False):
            raise ValueError(
                f"{type(gmm).__name__} does not fill bg_prob, and the seed for "
                "the graph cut is derived from it. Driven by such a backend "
                "this pipeline would silently label every pixel probable "
                "foreground and segment a field of zeros.")
        self.gmm      = gmm
        self._max_iter = max(PUSH_RELABEL_MIN_ITER,
                             PUSH_RELABEL_ITER_PER_DIAMETER * (gmm.height + gmm.width))
        self._warned_no_convergence = False
        self.roi_rect = roi_rect
        self.gamma    = float(gamma)
        H, W = gmm.height, gmm.width
        self._H, self._W = H, W
        N = H * W

        # ROI binary mask: 255 inside rectangle, 0 outside (60%×70% as in cv2_grabcut2.py)
        self._roi_mask = np.zeros((H, W), dtype=np.uint8)
        rx, ry, rw, rh = roi_rect
        self._roi_mask[ry:ry + rh, rx:rx + rw] = np.uint8(255)

        # GC label map (0/1/2/3) — rebuilt each frame
        self._gc_mask = np.zeros((H, W), dtype=np.uint8)

        # Dual spatial GMMs (Rother 2004)
        self._bg_gmm = GrabCutGMM()
        self._fg_gmm = GrabCutGMM()

        # Neg-log-likelihood maps per GMM (float32, H×W)
        self._nlp_bg = np.zeros((H, W), dtype=np.float32)
        self._nlp_fg = np.zeros((H, W), dtype=np.float32)

        # Graph cut capacity arrays (flat N)
        self._cap_src   = np.zeros(N, dtype=np.float32)
        self._cap_snk   = np.zeros(N, dtype=np.float32)
        self._cap_right = np.zeros(N, dtype=np.float32)
        self._cap_down  = np.zeros(N, dtype=np.float32)

        # N-weight scratch (float64 — calc_nweights_numba signature)
        self._leftW    = np.zeros((H, W), dtype=np.float64)
        self._upleftW  = np.zeros((H, W), dtype=np.float64)
        self._upW      = np.zeros((H, W), dtype=np.float64)
        self._uprightW = np.zeros((H, W), dtype=np.float64)

        # Output masks + morph scratch
        self._final_mask = np.zeros((H, W), dtype=np.uint8)
        self._morph_tmp1 = np.zeros((H, W), dtype=np.uint8)
        self._morph_tmp2 = np.zeros((H, W), dtype=np.uint8)

        # Blurred-background composite
        self._blurred    = np.zeros((H, W, 3), dtype=np.uint8)
        self._composite  = np.zeros((H, W, 3), dtype=np.uint8)
        self._blur_ksize = blur_ksize | 1   # must be odd

        # Warm up all Numba kernels
        warmup_push_relabel()
        warmup_morph()
        warmup_fgd_gmm()

    def rqstep(self, frame_bgr, update_alpha=-1.0):
        """Run one full pipeline frame.

        Parameters
        ----------
        frame_bgr    : (H, W, 3) uint8
        update_alpha : float  (negative = MOG2 auto-ramp)

        Returns
        -------
        mog2_mask   : (H, W) uint8   — raw MOG2 result
        bg_prob     : (H, W) float32 — MOG2 background probability
        final_mask  : (H, W) uint8   — segmentation mask (255=fg, 0=bg)
        composite   : (H, W, 3) uint8 — fg sharp, bg blurred
        elapsed_mog2: float           — MOG2 wall time (seconds)
        """
        import cv2
        H, W = self._H, self._W

        # ── Stage 1: MOG2 ────────────────────────────────────────────────────
        planar = to_planar(frame_bgr)
        # step() returns (mask, seconds) for every backend; the background
        # confidence rides along on the model as an attribute so adding it did
        # not change a contract that four backends and five callers depend on.
        mog2_mask, elapsed_mog2 = self.gmm.step(planar, update_alpha)
        bg_prob = self.gmm.bg_prob

        # ── Stage 2: GC label map from MOG2 bg_prob + ROI ────────────────────
        _make_gc_mask(bg_prob, self._roi_mask, self._gc_mask,
                      BG_HARD_THRESH, FG_HARD_THRESH)

        # ── Stage 3: fit dual spatial GMMs ───────────────────────────────────
        img_f64 = frame_bgr.astype(np.float64)
        self._bg_gmm.fit(img_f64, self._gc_mask, is_fg=False)
        self._fg_gmm.fit(img_f64, self._gc_mask, is_fg=True)

        # ── Stage 4: evaluate GMM neg-log-likelihoods ────────────────────────
        self._bg_gmm.neg_log_prob(img_f64, self._nlp_bg)
        self._fg_gmm.neg_log_prob(img_f64, self._nlp_fg)

        # ── Stage 5: image contrast beta + n-weights ─────────────────────────
        beta = calc_beta_numba(img_f64)
        calc_nweights_numba(img_f64, beta, self.gamma,
                            self._leftW, self._upleftW,
                            self._upW, self._uprightW)

        # Dynamic lambda: 9 × max n-link weight (Rother 2004)
        max_nw = max(float(self._leftW.max()), float(self._upW.max()))
        lam = max_nw * LAM_FACTOR if max_nw > 0.0 else self.gamma * LAM_FACTOR

        # ── Stage 6: build graph capacity arrays ─────────────────────────────
        _build_tlinks(self._gc_mask, self._nlp_bg, self._nlp_fg, lam,
                      self._cap_src, self._cap_snk)
        _build_nlinks(self._leftW.astype(np.float32),
                      self._upW.astype(np.float32),
                      H, W, self._cap_right, self._cap_down)

        # ── Stage 7: Push-Relabel ────────────────────────────────────────────
        labeling, iterations = push_relabel(
            self._cap_src, self._cap_snk,
            self._cap_right, self._cap_down,
            H, W,
            self._max_iter, PUSH_RELABEL_RELABEL_FREQ,
        )
        if iterations >= self._max_iter:
            # Not a minimal cut. Say so once rather than emit a plausible mask.
            if not self._warned_no_convergence:
                warnings.warn(
                    f"push_relabel hit its {self._max_iter}-iteration cap at "
                    f"{W}x{H}; the cut is not minimal. Raise "
                    f"PUSH_RELABEL_ITER_PER_DIAMETER.", RuntimeWarning, stacklevel=2)
                self._warned_no_convergence = True

        # Our PR returns 1 for SINK-reachable (background side).
        # Invert so that foreground pixels = 255.
        fg = (labeling == 0).reshape(H, W).astype(np.uint8)
        np.multiply(fg, np.uint8(255), out=self._final_mask)

        # ── Stage 8: morphological cleanup ───────────────────────────────────
        morphological_close(self._final_mask, self._morph_tmp1, self._morph_tmp2,
                            H, W, radius=3)
        morphological_open(self._morph_tmp2, self._morph_tmp1, self._final_mask,
                           H, W, radius=2)
        clean = largest_component(self._final_mask, H, W)
        np.copyto(self._final_mask, clean)

        # ── Stage 9: blurred-background composite ────────────────────────────
        import cv2 as _cv2
        ks = self._blur_ksize
        self._blurred[:] = _cv2.GaussianBlur(frame_bgr, (ks, ks), 0)
        _compose_blur(frame_bgr, self._blurred, self._final_mask, self._composite)

        return mog2_mask, bg_prob, self._final_mask, self._composite, elapsed_mog2
