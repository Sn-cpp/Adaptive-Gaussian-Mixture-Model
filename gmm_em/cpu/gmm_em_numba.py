import numpy as np
from numba import njit, prange

from gmm_em.gmm_em_common import GMM_EM_Base

class GMM_EM_Numba_CPU(GMM_EM_Base):
    """Per-frame full-covariance 5-component GMM for one class (bg or fg).

    State: model (K,13), inv_covs (K,3,3), cov_dets (K,).
    Call `fit(frame_bgr, mask_gc, is_fg=True/False)` each frame.
    Call `neg_log_prob(frame_bgr, out)` to fill the t-link capacity array.
    """

    def __init__(self, height, width):
        super().__init__(height, width)

    def fit(self, frame: np.ndarray, label_map: np.ndarray, is_fg: bool):
        """Refit GMM from pixels of this class in the current frame.

        frame         : (H, W, 3) float32
        label_map     : (H, W) uint8  — GC label map (0/1/2/3)
        is_fg         : bool
        """

        # Step 1: assign each pixel to best component (E-step)
        _assign_components(frame, label_map,
                           self.model, self.inv_covs, self.cov_dets,
                           self._comp)

        # Step 2: accumulate sufficient statistics (M-step numerators)
        _accumulate_stats(frame, label_map, self._comp, is_fg,
                          self._sums, self._prods, self._counts, self._total)

        total = int(self._total[0])
        if total == 0:
            return

        # Step 3: compute means, covariances, coefs; cache inv + det
        for ci in range(self.K):
            n = int(self._counts[ci])
            if n == 0:
                self.model[ci, 0] = 0.0
                self.cov_dets[ci] = 0.0
                continue

            self.model[ci, 0] = n / total

            inv_n = 1.0 / n
            mb = self._sums[ci, 0] * inv_n
            mg = self._sums[ci, 1] * inv_n
            mr = self._sums[ci, 2] * inv_n
            self.model[ci, 1] = mb
            self.model[ci, 2] = mg
            self.model[ci, 3] = mr

            # Cov = E[xx'] - μμ'  (stored in model[ci, 4:13])
            c = self.model[ci, 4:13]
            c[0] = self._prods[ci, 0, 0] * inv_n - mb * mb
            c[1] = self._prods[ci, 0, 1] * inv_n - mb * mg
            c[2] = self._prods[ci, 0, 2] * inv_n - mb * mr
            c[3] = self._prods[ci, 1, 0] * inv_n - mg * mb
            c[4] = self._prods[ci, 1, 1] * inv_n - mg * mg
            c[5] = self._prods[ci, 1, 2] * inv_n - mg * mr
            c[6] = self._prods[ci, 2, 0] * inv_n - mr * mb
            c[7] = self._prods[ci, 2, 1] * inv_n - mr * mg
            c[8] = self._prods[ci, 2, 2] * inv_n - mr * mr

            det, inv = _cov_inv_det(c, 0.01)  # singular_fix=0.01 matches grabcut.cpp
            self.cov_dets[ci] = det
            self.inv_covs[ci] = inv

    def neg_log_prob(self, frame_bgr_f32, out):
        """-log P(color | GMM) for every pixel → written into `out` (H,W) float32."""
        _eval_gmm(frame_bgr_f32, self.model, self.inv_covs, self.cov_dets, out)


# ── Kernel: compute inverse + determinant for one 3×3 symmetric matrix ──────
@njit(cache=True)
def _cov_inv_det(c, singular_fix):
    """In-place: compute det and 3×3 inverse of covariance `c` (3×3 row-major).

    `c` is a (9,) float32 view.  Returns (det, inv_3x3).
    Adds `singular_fix` to diagonal if det ≤ 1e-6 (matches grabcut.cpp).
    """
    c00 = c[0]; c01 = c[1]; c02 = c[2]
    c10 = c[3]; c11 = c[4]; c12 = c[5]
    c20 = c[6]; c21 = c[7]; c22 = c[8]

    det = (c00 * (c11 * c22 - c12 * c21)
           - c01 * (c10 * c22 - c12 * c20)
           + c02 * (c10 * c21 - c11 * c20))

    if det <= 1e-6 and singular_fix > 0.0:
        c[0] += singular_fix; c[4] += singular_fix; c[8] += singular_fix
        c00 = c[0]; c11 = c[4]; c22 = c[8]
        det = (c00 * (c11 * c22 - c12 * c21)
               - c01 * (c10 * c22 - c12 * c20)
               + c02 * (c10 * c21 - c11 * c20))

    inv = np.empty((3, 3), dtype=np.float32)
    if det > 1e-300:
        inv_d = 1.0 / det
        inv[0, 0] =  (c11 * c22 - c12 * c21) * inv_d
        inv[1, 0] = -(c10 * c22 - c12 * c20) * inv_d
        inv[2, 0] =  (c10 * c21 - c11 * c20) * inv_d
        inv[0, 1] = -(c01 * c22 - c02 * c21) * inv_d
        inv[1, 1] =  (c00 * c22 - c02 * c20) * inv_d
        inv[2, 1] = -(c00 * c21 - c01 * c20) * inv_d
        inv[0, 2] =  (c01 * c12 - c02 * c11) * inv_d
        inv[1, 2] = -(c00 * c12 - c02 * c10) * inv_d
        inv[2, 2] =  (c00 * c11 - c01 * c10) * inv_d
    else:
        for i in range(3):
            for j in range(3):
                inv[i, j] = 0.0
    return det, inv

# ── Kernel: assign each pixel to the nearest component ──────────────────────
@njit(parallel=True, cache=True)
def _assign_components(frame_bgr, mask_gc, model, inv_covs, cov_dets, comp_idx):
    """For each pixel write its best-fitting component index into comp_idx.

    mask_gc : (H, W) uint8  — 0=GC_BGD, 1=GC_FGD, 2=GC_PR_BGD, 3=GC_PR_FGD
    comp_idx: (H, W) int32  — written in-place
    """
    H = frame_bgr.shape[0]
    W = frame_bgr.shape[1]
    K = model.shape[0]
    eps = 1e-300

    for y in prange(H):
        for x in range(W):
            b = frame_bgr[y, x, 0]
            g = frame_bgr[y, x, 1]
            r = frame_bgr[y, x, 2]

            best_ci = np.int32(0)
            best_p  = np.float32(-1.0)
            for ci in range(K):
                coef = model[ci, 0]
                if coef <= 0.0:
                    continue
                det = cov_dets[ci]
                if det <= eps:
                    continue

                db = b - model[ci, 1]
                dg = g - model[ci, 2]
                dr = r - model[ci, 3]

                ic = inv_covs[ci]
                mult = (db * (db * ic[0, 0] + dg * ic[1, 0] + dr * ic[2, 0])
                      + dg * (db * ic[0, 1] + dg * ic[1, 1] + dr * ic[2, 1])
                      + dr * (db * ic[0, 2] + dg * ic[1, 2] + dr * ic[2, 2]))

                p = coef / (det ** 0.5 + eps) * np.exp(-0.5 * mult)
                if p > best_p:
                    best_p  = p
                    best_ci = np.int32(ci)

            comp_idx[y, x] = best_ci

# ── Kernel: accumulate sufficient statistics per component ───────────────────
@njit(cache=True)
def _accumulate_stats(frame_bgr, mask_gc, comp_idx, is_fg,
                      sums, prods, counts, total):
    """Accumulate mean/cov sufficient statistics for pixels belonging to this GMM.

    is_fg : bool — True = accumulate pixels where mask is FGD/PR_FGD,
                   False = accumulate pixels where mask is BGD/PR_BGD
    sums  : (K, 3) float32   — colour sums per component
    prods : (K, 3, 3) float32 — outer-product sums per component
    counts: (K,) int64
    total[0]: int64 — total pixel count (scalar in length-1 array)
    """
    H = frame_bgr.shape[0]
    W = frame_bgr.shape[1]
    K = sums.shape[0]

    for ci in range(K):
        sums[ci, 0] = 0.0; sums[ci, 1] = 0.0; sums[ci, 2] = 0.0
        for i in range(3):
            for j in range(3):
                prods[ci, i, j] = 0.0
        counts[ci] = np.int64(0)
    total[0] = np.int64(0)

    for y in range(H):
        for x in range(W):
            m = mask_gc[y, x]
            # 0=GC_BGD, 1=GC_FGD, 2=GC_PR_BGD, 3=GC_PR_FGD
            pixel_is_fg = (m == np.uint8(1)) or (m == np.uint8(3))
            if pixel_is_fg != is_fg:
                continue

            ci = comp_idx[y, x]
            b  = frame_bgr[y, x, 0]
            g  = frame_bgr[y, x, 1]
            r  = frame_bgr[y, x, 2]

            sums[ci, 0] += b; sums[ci, 1] += g; sums[ci, 2] += r
            prods[ci, 0, 0] += b*b; prods[ci, 0, 1] += b*g; prods[ci, 0, 2] += b*r
            prods[ci, 1, 0] += g*b; prods[ci, 1, 1] += g*g; prods[ci, 1, 2] += g*r
            prods[ci, 2, 0] += r*b; prods[ci, 2, 1] += r*g; prods[ci, 2, 2] += r*r
            counts[ci] += np.int64(1)
            total[0]   += np.int64(1)

# ── Kernel: evaluate GMM log-likelihood for every pixel ─────────────────────
@njit(parallel=True, cache=True)
def _eval_gmm(frame_bgr, model, inv_covs, cov_dets, out):
    """Compute -log P(color | GMM) for every pixel, written into `out` (H,W).

    frame_bgr : (H, W, 3) float32
    model     : (K, 13) float32
    inv_covs  : (K, 3, 3) float32
    cov_dets  : (K,) float32
    out       : (H, W) float32  — written in-place
    """
    H = frame_bgr.shape[0]
    W = frame_bgr.shape[1]
    K = model.shape[0]

    eps = 1e-300
    for y in prange(H):
        for x in range(W):
            b = frame_bgr[y, x, 0]
            g = frame_bgr[y, x, 1]
            r = frame_bgr[y, x, 2]

            prob = np.float32(0.0)
            for ci in range(K):
                coef = model[ci, 0]
                if coef <= 0.0:
                    continue
                det = cov_dets[ci]
                if det <= eps:
                    continue

                db = b - model[ci, 1]
                dg = g - model[ci, 2]
                dr = r - model[ci, 3]

                ic = inv_covs[ci]
                mult = (db * (db * ic[0, 0] + dg * ic[1, 0] + dr * ic[2, 0])
                      + dg * (db * ic[0, 1] + dg * ic[1, 1] + dr * ic[2, 1])
                      + dr * (db * ic[0, 2] + dg * ic[1, 2] + dr * ic[2, 2]))

                prob += coef / (det ** 0.5 + eps) * np.exp(-0.5 * mult)

            out[y, x] = np.float32(-np.log(prob + eps))

def warmup_em_gmm_jit(H=4, W=4, K=2):
    """Pre-compile all Numba kernels on tiny dummy data."""
    frame = np.zeros((H, W, 3), dtype=np.float32)
    mask  = np.zeros((H, W),    dtype=np.uint8)
    mask[H // 2:, :] = np.uint8(3)  # some PR_FGD pixels

    model    = np.zeros((K, 13),    dtype=np.float32)
    inv_covs = np.zeros((K, 3, 3),  dtype=np.float32)
    cov_dets = np.zeros(K,          dtype=np.float32)
    comp_idx = np.zeros((H, W),     dtype=np.int32)
    out      = np.zeros((H, W),     dtype=np.float32)

    for ci in range(K):
        model[ci, 0] = 1.0 / K
        model[ci, 4] = 1.0   # diagonal cov
        model[ci, 8] = 1.0
        model[ci, 12] = 1.0
        # seed inv and det
        for d in range(3):
            inv_covs[ci, d, d] = 1.0
        cov_dets[ci] = 1.0

    _assign_components(frame, mask, model, inv_covs, cov_dets, comp_idx)

    sums   = np.zeros((K, 3),    dtype=np.float32)
    prods  = np.zeros((K, 3, 3), dtype=np.float32)
    counts = np.zeros(K,         dtype=np.int64)
    total  = np.zeros(1,         dtype=np.int64)
    _accumulate_stats(frame, mask, comp_idx, True,
                      sums, prods, counts, total)

    _eval_gmm(frame, model, inv_covs, cov_dets, out)