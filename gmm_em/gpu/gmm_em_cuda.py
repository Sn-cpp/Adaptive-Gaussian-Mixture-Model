import numpy as np
from numba import cuda

from gmm_em.gmm_em_common import GMM_EM_Base
from gmm_em.cpu.gmm_em_numba import _cov_inv_det    # reuse CPU 3×3 inv/det


class GMM_EM_CUDA(GMM_EM_Base):
    """Full-covariance K-component GMM — CUDA-accelerated E and scoring steps.

    fit() layout
    ──────────────
    E-step  (_assign_components_cuda)  : GPU — one thread per pixel
    M-step accumulate (_accumulate_stats_cuda): GPU — one thread per pixel,
                                                atomic adds into K buckets
    M-step solve (covariance inversion): CPU — K=5 iterations, negligible

    neg_log_prob() (_eval_gmm_cuda)    : GPU — one thread per pixel

    Device-side state
    ──────────────────
    d_model    : (K, 13) float32
    d_inv_covs : (K, 3, 3) float32
    d_cov_dets : (K,) float32
    d_comp     : (H, W) int32          — component assignment scratch
    d_sums     : (K, 3) float32        — atomic accumulation buffers
    d_prods    : (K, 9) float32        — (K, 3×3 flattened)
    d_counts   : (K,) int32

    The host-side `model`, `inv_covs`, `cov_dets` in GMM_EM_Base are kept
    in sync after every fit() so that any CPU code reading them still works.
    """

    def __init__(self, height: int, width: int):
        super().__init__(height, width)

        # Upload base-class host arrays to device
        self.d_model    = cuda.to_device(self.model)
        self.d_inv_covs = cuda.to_device(self.inv_covs)
        self.d_cov_dets = cuda.to_device(self.cov_dets)

        # Per-frame scratch — stays on device between fit() and neg_log_prob()
        self.d_comp   = cuda.to_device(self._comp)
        self.d_sums   = cuda.to_device(np.zeros((self.K, 3), np.float32))
        self.d_prods  = cuda.to_device(np.zeros((self.K, 9), np.float32))  # 3×3 flat
        self.d_counts = cuda.to_device(np.zeros(self.K, np.int32))

        self._BLOCK = 256
        self._grid  = (height * width + self._BLOCK - 1) // self._BLOCK

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, d_frame, d_mask, is_fg: bool):
        """Refit GMM from pixels of this class in the current frame.

        d_frame : (H, W, 3) float32  device array
        d_mask  : (H, W) uint8       device array  (GC labels 0/1/2/3)
        is_fg   : bool
        """
        K       = np.int32(self.K)
        H       = np.int32(self.H)
        W       = np.int32(self.W)
        fg_flag = np.int32(1 if is_fg else 0)

        # E-step: assign every pixel to its best-fitting component
        _assign_components_cuda[self._grid, self._BLOCK](
            d_frame, d_mask, self.d_model, self.d_inv_covs, self.d_cov_dets,
            self.d_comp, H, W, K)

        # M-step accumulate: reset buffers then atomic-add per pixel
        k_grid = (self.K + self._BLOCK - 1) // self._BLOCK
        _zero_stats_cuda[k_grid, self._BLOCK](
            self.d_sums, self.d_prods, self.d_counts, K)
        _accumulate_stats_cuda[self._grid, self._BLOCK](
            d_frame, d_mask, self.d_comp, fg_flag,
            self.d_sums, self.d_prods, self.d_counts,
            H, W, K)
        cuda.synchronize()

        # M-step solve: tiny K-loop on CPU (K=5, negligible)
        h_sums   = self.d_sums.copy_to_host()
        h_prods  = self.d_prods.copy_to_host().reshape(self.K, 3, 3)
        h_counts = self.d_counts.copy_to_host().astype(np.int64)
        total    = int(h_counts.sum())
        if total == 0:
            return

        for ci in range(self.K):
            n = int(h_counts[ci])
            if n == 0:
                self.model[ci, 0] = 0.0
                self.cov_dets[ci] = 0.0
                continue

            self.model[ci, 0] = n / total
            inv_n = 1.0 / n
            mb = h_sums[ci, 0] * inv_n
            mg = h_sums[ci, 1] * inv_n
            mr = h_sums[ci, 2] * inv_n
            self.model[ci, 1] = mb
            self.model[ci, 2] = mg
            self.model[ci, 3] = mr

            c = self.model[ci, 4:13]
            c[0] = h_prods[ci, 0, 0] * inv_n - mb * mb
            c[1] = h_prods[ci, 0, 1] * inv_n - mb * mg
            c[2] = h_prods[ci, 0, 2] * inv_n - mb * mr
            c[3] = h_prods[ci, 1, 0] * inv_n - mg * mb
            c[4] = h_prods[ci, 1, 1] * inv_n - mg * mg
            c[5] = h_prods[ci, 1, 2] * inv_n - mg * mr
            c[6] = h_prods[ci, 2, 0] * inv_n - mr * mb
            c[7] = h_prods[ci, 2, 1] * inv_n - mr * mg
            c[8] = h_prods[ci, 2, 2] * inv_n - mr * mr

            det, inv = _cov_inv_det(c, 0.01)
            self.cov_dets[ci] = det
            self.inv_covs[ci] = inv

        # Push updated model back to device
        self.d_model.copy_to_device(self.model)
        self.d_inv_covs.copy_to_device(self.inv_covs)
        self.d_cov_dets.copy_to_device(self.cov_dets)

    def neg_log_prob(self, d_frame, d_out):
        """-log P(color | GMM) for every pixel → written into d_out (H,W) float32."""
        _eval_gmm_cuda[self._grid, self._BLOCK](
            d_frame, self.d_model, self.d_inv_covs, self.d_cov_dets,
            d_out, np.int32(self.H), np.int32(self.W), np.int32(self.K))


# ── CUDA device helpers ───────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _expf(x):
    return cuda.libdevice.expf(x)

@cuda.jit(device=True, inline=True)
def _logf(x):
    return cuda.libdevice.logf(x)


# ── CUDA kernels ──────────────────────────────────────────────────────────────

@cuda.jit
def _assign_components_cuda(frame, mask, model, inv_covs, cov_dets,
                             comp, H, W, K):
    """E-step: assign each pixel to the best-fitting GMM component."""
    idx = cuda.grid(1)
    if idx >= H * W:
        return
    y = idx // W
    x = idx % W

    b = frame[y, x, 0]
    g = frame[y, x, 1]
    r = frame[y, x, 2]

    best_ci = np.int32(0)
    best_p  = np.float32(-1.0)
    eps     = np.float32(1e-30)

    for ci in range(K):
        coef = model[ci, 0]
        if coef <= np.float32(0.0):
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

        p = coef / (det ** np.float32(0.5) + eps) * _expf(-np.float32(0.5) * mult)
        if p > best_p:
            best_p  = p
            best_ci = np.int32(ci)

    comp[y, x] = best_ci


@cuda.jit
def _zero_stats_cuda(sums, prods, counts, K):
    """Reset accumulation buffers (one thread per component)."""
    i = cuda.grid(1)
    if i >= K:
        return
    sums[i, 0] = np.float32(0.0)
    sums[i, 1] = np.float32(0.0)
    sums[i, 2] = np.float32(0.0)
    counts[i]  = np.int32(0)
    for j in range(9):
        prods[i, j] = np.float32(0.0)


@cuda.jit
def _accumulate_stats_cuda(frame, mask, comp, fg_flag,
                            sums, prods, counts,
                            H, W, K):
    """M-step accumulate: atomic adds of colour stats per component."""
    idx = cuda.grid(1)
    if idx >= H * W:
        return
    y = idx // W
    x = idx % W

    m = mask[y, x]
    pixel_is_fg = np.int32(1) if (m == np.uint8(1) or m == np.uint8(3)) else np.int32(0)
    if pixel_is_fg != fg_flag:
        return

    ci = comp[y, x]
    b  = frame[y, x, 0]
    g  = frame[y, x, 1]
    r  = frame[y, x, 2]

    cuda.atomic.add(sums,   (ci, 0), b)
    cuda.atomic.add(sums,   (ci, 1), g)
    cuda.atomic.add(sums,   (ci, 2), r)
    cuda.atomic.add(counts, ci,      np.int32(1))

    cuda.atomic.add(prods, (ci, 0), b * b)
    cuda.atomic.add(prods, (ci, 1), b * g)
    cuda.atomic.add(prods, (ci, 2), b * r)
    cuda.atomic.add(prods, (ci, 3), g * b)
    cuda.atomic.add(prods, (ci, 4), g * g)
    cuda.atomic.add(prods, (ci, 5), g * r)
    cuda.atomic.add(prods, (ci, 6), r * b)
    cuda.atomic.add(prods, (ci, 7), r * g)
    cuda.atomic.add(prods, (ci, 8), r * r)


@cuda.jit
def _eval_gmm_cuda(frame, model, inv_covs, cov_dets, out, H, W, K):
    """-log P(color | GMM) for every pixel, written into out (H,W)."""
    idx = cuda.grid(1)
    if idx >= H * W:
        return
    y = idx // W
    x = idx % W

    b = frame[y, x, 0]
    g = frame[y, x, 1]
    r = frame[y, x, 2]

    eps  = np.float32(1e-30)
    prob = np.float32(0.0)

    for ci in range(K):
        coef = model[ci, 0]
        if coef <= np.float32(0.0):
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

        prob += coef / (det ** np.float32(0.5) + eps) * _expf(-np.float32(0.5) * mult)

    out[y, x] = np.float32(-_logf(prob + eps))


def warmup_gmm_em_cuda_jit(H=4, W=4):
    """Pre-compile all CUDA GMM kernels on tiny dummy data."""
    gmm    = GMM_EM_CUDA(H, W)
    d_frame = cuda.to_device(np.zeros((H, W, 3), dtype=np.float32))
    d_mask  = cuda.to_device(np.zeros((H, W),    dtype=np.uint8))
    d_out   = cuda.to_device(np.zeros((H, W),    dtype=np.float32))
    gmm.fit(d_frame, d_mask, is_fg=False)
    gmm.neg_log_prob(d_frame, d_out)
    cuda.synchronize()
