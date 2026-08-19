import numpy as np
from numba import cuda, float32, int32, uint8

from gmm_mask.gmm_mask_common import GMM_Mask_Base, kernel_args
from gmm_mask.gpu import blur_kernels as bk
from gmm_mask.gpu import post_kernels as pk
from settings import (
    BLUR_KSIZE,
    BLUR_SIGMA,
    FLT_EPSILON,
    MOG2_BG_PROB_THRESHOLD,
    MOG2_HISTORY,
)

TILE_X = 32
TILE_Y = 8
MAX_C  = 3

# ── Adaptive-alpha constants (SuBSENSE-inspired) ──────────────────────────────
_ALPHA_DECAY   = float32(0.98)   # FG pixel: alpha *= this  (slows absorption)
_ALPHA_RESTORE = float32(1.02)   # BG pixel: alpha *= this  (restores toward base)

# ── Neighbor-propagation constants (ViBe-inspired) ────────────────────────────
# With probability 1/PHI a background pixel copies its best Gaussian into a
# random 8-neighbor's lowest-weight slot.  PHI=16 matches ViBe's default.
_PHI = int32(16)

# 8-neighbor offsets stored as flat (dy, dx) pairs
_NBR_DY = ( 0,  0, -1,  1, -1, -1,  1,  1)
_NBR_DX = (-1,  1,  0,  0, -1,  1, -1,  1)


class GMM_Mask_CUDA_v1(GMM_Mask_Base):
    """v1 — post-processing moved onto the GPU, plus two optional model extras.

    **The v1 contribution is the post-processing chain.** v0 computes the mask
    on the device and then hands it to OpenCV on the host: threshold, median,
    fill. Every one of those but the fill is a per-pixel or small-stencil
    operation with no reason to be on the CPU, and each one costs a full
    round trip of the mask. v1 keeps the mask resident and runs
    `threshold -> median5` as kernels, so exactly one H2D (the frame) and one
    D2H (the refined mask) cross the bus per frame.

    `fill_holes` deliberately stays on the host — see `post_kernels`.

    The two model extras below are **off by default**. They were written for
    webcam footage and have never been scored on the car dataset; adaptive
    alpha in particular slows absorption, which is what you want for a person
    who sits still and precisely what you do not want behind a moving car.
    Turn them on only with a number from `eval_highway.py` to justify it.

    1. `adaptive_alpha` — per-pixel learning rate (SuBSENSE feedback).
       Foreground pixels decay their alpha so stationary subjects are absorbed
       far more slowly than with a global alpha.

    2. `propagate` — spatial Gaussian neighbour propagation (ViBe coherence).
       Background pixels occasionally copy their best Gaussian into a random
       8-neighbour, bridging the gaps that shatter a mask.
    """

    def __init__(self, height: int, width: int, *args,
                 adaptive_alpha: bool = False, propagate: bool = False,
                 bg_prob_threshold: float = MOG2_BG_PROB_THRESHOLD,
                 post: bool = True, colorspace: str = "ycrcb", **kwargs):
        super().__init__(height, width, *args, **kwargs)
        self.adaptive_alpha = bool(adaptive_alpha)
        self.propagate = bool(propagate)
        self.post = bool(post)
        self.to_ycrcb = (colorspace == "ycrcb")
        self.bg_prob_threshold = np.float32(bg_prob_threshold)

        # Per-pixel alpha map — initialised to the same warm-up value that
        # GMM_Mask_Base.learning_rate() would return for frame 1.
        base_alpha = np.float32(1.0 / min(2, MOG2_HISTORY))
        self._alpha_map = np.full((height, width), base_alpha, dtype=np.float32)

        self.d_means    = cuda.to_device(self.means)
        self.d_vars     = cuda.to_device(self.vars)
        self.d_weights  = cuda.to_device(self.weights)
        self.d_modes    = cuda.to_device(self.modes)
        self.d_mask     = cuda.to_device(self.mask)
        self.d_bg_prob  = cuda.to_device(self.bg_prob)
        self.d_alpha_map = cuda.to_device(self._alpha_map)

        # Post-processing scratch. Allocated once: a per-frame allocation would
        # cost more than the kernels it feeds.
        self.refined = np.zeros((height, width), dtype=np.uint8)
        self.d_post_a = cuda.device_array((height, width), dtype=np.uint8)
        self.d_post_b = cuda.device_array((height, width), dtype=np.uint8)

        self.block = (TILE_X, TILE_Y)
        self.grid  = (
            int((self.W + TILE_X - 1) // TILE_X),
            int((self.H + TILE_Y - 1) // TILE_Y),
        )

        # ── Kernel 2 buffers: the BGR frame is the only thing crossing the bus
        # Allocated once. The blur has its own launch geometry and must never
        # borrow `self.grid`/`self.block`: a radius-7 halo is 14 rows, so the
        # model's 32x8 block would leave half of every shared tile unloaded,
        # and the failure is silent — a plausible-looking but wrong blur.
        self.d_frame_bgr = cuda.device_array((height, width, 3), np.uint8)
        self.d_ycrcb     = cuda.device_array((3, height, width), np.float32)
        self.d_blur_tmp  = cuda.device_array((height, width, 3), np.uint16)
        self.d_out_bgr   = cuda.device_array((height, width, 3), np.uint8)
        self.out_bgr     = np.empty((height, width, 3), dtype=np.uint8)
        self.d_kq        = cuda.to_device(
            bk.gaussian_kernel_q8(BLUR_KSIZE, BLUR_SIGMA))
        self.blur_block  = (bk.BLUR_TILE_X, bk.BLUR_TILE_Y)
        self.blur_grid   = bk.blur_grid_for(height, width)

        # Whatever `step_device` last returned — d_post_b with post on, d_mask
        # with it off. `composite()` reads this rather than naming a buffer,
        # because naming one is how `--no-fill` ends up compositing against
        # MOG2's raw decision instead of the refined mask.
        self._d_mask_out = None

    # ── public API (mirrors GMM_Mask_CUDA) ────────────────────────────────────

    def step_device(self, d_frame, args, stream=0):
        """Enqueue the model kernel and the post chain; no synchronisation.

        Returns (d_refined, d_bg_prob) when post-processing is on, so callers
        get the mask they should actually composite with. `d_mask` still holds
        MOG2's own binary decision for anyone comparing against OpenCV.
        """
        mog2_step_v1_kernel[self.grid, self.block, stream](
            d_frame,
            self.d_weights, self.d_means, self.d_vars,
            self.d_modes, self.d_mask, self.d_bg_prob,
            self.d_alpha_map,
            self.adaptive_alpha,
            *args,
        )
        if self.propagate:
            propagate_kernel[self.grid, self.block, stream](
                self.d_weights, self.d_means, self.d_vars,
                self.d_modes, self.d_mask,
                self.d_alpha_map,   # passed for the base-alpha clamp only
            )
        if not self.post:
            return self.d_mask, self.d_bg_prob

        pk.threshold_kernel[self.grid, self.block, stream](
            self.d_bg_prob, self.d_post_a, self.bg_prob_threshold)
        pk.median5_kernel[self.grid, self.block, stream](
            self.d_post_a, self.d_post_b)
        return self.d_post_b, self.d_bg_prob

    def _step_kernel(self, frame, to_host, args):
        d_frame = cuda.to_device(frame)
        d_out, _ = self.step_device(d_frame, args)
        cuda.synchronize()

        if not to_host:
            return d_out, self.d_bg_prob

        # One D2H for the mask the caller will composite with. bg_prob is not
        # copied when post-processing is on: the GPU already thresholded it, so
        # the transfer is pure waste — 8 MB a frame at 1080p, a third of the
        # frame budget. Return None rather than the stale buffer, because a
        # caller reading a silently-zero confidence map gets a mask that is
        # entirely foreground and no error to explain it.
        if self.post:
            d_out.copy_to_host(self.refined)
            return self.refined, None
        self.d_mask.copy_to_host(self.mask)
        self.d_bg_prob.copy_to_host(self.bg_prob)
        return self.mask, self.bg_prob

    # ── Kernel 2: BGR in, composite out, one conversion on the device ────────

    def mask_from_bgr(self, frame_bgr, update_alpha=-1.0, to_host=True):
        """Upload a BGR uint8 frame and return the refined mask.

        This is the ingest half of the pipeline `main.py` uses. It replaces the
        host's `cvtColor` + `astype(float32)` + transpose + 12 byte/pixel
        upload with a 3 byte/pixel upload and a conversion kernel: 24.88 MB
        becomes 6.22 MB per frame at 1080p, and a 25 MB numpy shuffle
        disappears from the host entirely.

        The conversion is bit-exact with `cv2.cvtColor`, so the model sees
        byte-identical input to the old path and the scored mask is unchanged.
        That is not an argument, it is a test — `test_blur.py` pins the kernel
        against cv2, and `bench_t4.py --only equivalence` pins the resulting
        mask across 120 frames at 1080p — measured, zero pixels differ.
        `eval_highway.py --parity-vs` extends that to the 1231 scored CDnet
        frames wherever the dataset is available.

        `to_host=False` leaves the mask on the device and returns the device
        array. That is the `--no-fill` path: the mask never has to come down
        and go back up, saving both of its transfers.
        """
        self.d_frame_bgr.copy_to_device(np.ascontiguousarray(frame_bgr))
        bk.bgr2ycrcb_planar_kernel[self.blur_grid, self.blur_block](
            self.d_frame_bgr, self.d_ycrcb, self.to_ycrcb)

        # next_args() advances nframes and the alpha ramp; calling it once per
        # frame here is what keeps this path's learning rate identical to
        # apply()'s. Calling it twice, or not at all, desynchronises the model
        # from the reference without changing anything visible for many frames.
        d_out, _ = self.step_device(self.d_ycrcb, self.next_args(update_alpha))
        cuda.synchronize()
        self._d_mask_out = d_out

        if not to_host:
            return d_out
        d_out.copy_to_host(self.refined)
        return self.refined

    def _blur_kernels(self):
        """v1 runs the naive pair. v2 overrides this with the tiled pair."""
        return bk.blur_h_kernel, bk.blur_v_composite_kernel

    def composite(self, filled_mask=None):
        """Blur the background, keep the masked foreground, on the device.

        `filled_mask=None` means the mask never left the device — composite
        against whatever `mask_from_bgr` last produced. Otherwise the host
        filled the holes and the mask has to go back up; that round trip is the
        price of `fill_holes` staying sequential, and it is measured rather
        than hidden (see `bench_post.py`).
        """
        if filled_mask is None:
            if self._d_mask_out is None:
                raise RuntimeError("composite() before mask_from_bgr()")
            d_mask = self._d_mask_out
        else:
            self.d_post_a.copy_to_device(np.ascontiguousarray(filled_mask))
            d_mask = self.d_post_a

        kh, kv = self._blur_kernels()
        kh[self.blur_grid, self.blur_block](
            self.d_frame_bgr, self.d_blur_tmp, self.d_kq)
        kv[self.blur_grid, self.blur_block](
            self.d_blur_tmp, self.d_frame_bgr, d_mask, self.d_out_bgr,
            self.d_kq)
        cuda.synchronize()
        self.d_out_bgr.copy_to_host(self.out_bgr)
        return self.out_bgr

    def sync_state(self):
        self.d_weights.copy_to_host(self.weights)
        self.d_means.copy_to_host(self.means)
        self.d_vars.copy_to_host(self.vars)
        self.d_modes.copy_to_host(self.modes)
        self.d_alpha_map.copy_to_host(self._alpha_map)

    def background_image(self):
        self.sync_state()
        return super().background_image()


# ── Kernel 1: MOG2 step with per-pixel alpha ──────────────────────────────────

@cuda.jit(device=True)
def _detect_shadow(frame, y, x, C, nmodes, weights, means, vars_, Tb, TB, tau):
    t_weight = float32(0.0)
    for mode in range(nmodes):
        num = float32(0.0)
        den = float32(0.0)
        for c in range(C):
            num += frame[c, y, x] * means[mode, c, y, x]
            den += means[mode, c, y, x] * means[mode, c, y, x]
        if den == float32(0.0):
            return False
        if num <= den and num >= tau * den:
            a = num / den
            dist2a = float32(0.0)
            for c in range(C):
                dD = a * means[mode, c, y, x] - frame[c, y, x]
                dist2a += dD * dD
            if dist2a < Tb * vars_[mode, y, x] * a * a:
                return True
        t_weight += weights[mode, y, x]
        if t_weight > TB:
            return False
    return False


@cuda.jit
def mog2_step_v1_kernel(frame, weights, means, vars_, modes, mask, bg_prob,
                         alpha_map, adaptive_alpha,
                         alpha_base, prune_base, Tb, Tg, TB,
                         var_init, var_min, var_max,
                         tau, shadow_val, detect_shadows):
    """MOG2 step, optionally using a per-pixel learning rate from alpha_map.

    With `adaptive_alpha` off this is the plain MOG2 update and is bit-exact
    with `GMM_Mask_CUDA` — the flag has to leave that path untouched, or the
    parity claim against OpenCV goes with it.

    With it on, alpha_map supplies this pixel's alpha and prune is recomputed
    from it. After classification alpha_map is updated:
      - foreground pixel: alpha *= _ALPHA_DECAY   (slow down absorption)
      - background pixel: alpha *= _ALPHA_RESTORE (restore toward base)
    clamped to [alpha_base/8, alpha_base] so it never runs away.
    """
    x, y = cuda.grid(2)
    H = frame.shape[1]
    W = frame.shape[2]
    if y >= H or x >= W:
        return

    K = weights.shape[0]
    C = means.shape[1]

    # ── per-pixel alpha ───────────────────────────────────────────────────────
    if adaptive_alpha:
        alpha = alpha_map[y, x]
        prune = float32(-alpha * float32(0.05))   # CT = 0.05, mirrors kernel_args
    else:
        alpha = alpha_base
        prune = prune_base
    alpha1 = float32(1.0) - alpha

    dData = cuda.local.array(MAX_C, float32)

    background = False
    fits_pdf   = False
    nmodes     = int32(modes[y, x])
    total_weight   = float32(0.0)
    bg_weight_sum  = float32(0.0)

    mode = 0
    while mode < nmodes:
        weight    = alpha1 * weights[mode, y, x] + prune
        swap_count = 0

        if not fits_pdf:
            var   = vars_[mode, y, x]
            dist2 = float32(0.0)
            for c in range(C):
                dd = means[mode, c, y, x] - frame[c, y, x]
                dData[c] = dd
                dist2 += dd * dd

            if dist2 < Tb * var:
                bg_weight_sum += weights[mode, y, x]
                if total_weight < TB:
                    background = True

            if dist2 < Tg * var:
                fits_pdf = True
                weight  += alpha
                k        = alpha / weight
                for c in range(C):
                    means[mode, c, y, x] -= k * dData[c]
                varnew = var + k * (dist2 - var)
                if varnew < var_min:
                    varnew = var_min
                if varnew > var_max:
                    varnew = var_max
                vars_[mode, y, x] = varnew

                i = mode
                while i > 0:
                    if weight < weights[i - 1, y, x]:
                        break
                    swap_count += 1
                    tw = weights[i, y, x];        weights[i, y, x]     = weights[i-1, y, x]; weights[i-1, y, x] = tw
                    tv = vars_[i, y, x];          vars_[i, y, x]       = vars_[i-1, y, x];   vars_[i-1, y, x]   = tv
                    for c in range(C):
                        tm = means[i, c, y, x];   means[i, c, y, x]   = means[i-1, c, y, x]; means[i-1, c, y, x] = tm
                    i -= 1

        if weight < -prune:
            weight  = float32(0.0)
            nmodes -= 1

        weights[mode - swap_count, y, x] = weight
        total_weight += weight
        mode += 1

    # ── renormalise ───────────────────────────────────────────────────────────
    inv_weight = float32(0.0)
    if abs(total_weight) > FLT_EPSILON:
        inv_weight = float32(1.0) / total_weight
    for mode in range(nmodes):
        weights[mode, y, x] *= inv_weight

    bg_prob[y, x] = bg_weight_sum

    # ── new component if no fit ───────────────────────────────────────────────
    if not fits_pdf and alpha > float32(0.0):
        if nmodes == K:
            mode = K - 1
        else:
            mode   = nmodes
            nmodes += 1

        if nmodes == 1:
            weights[mode, y, x] = float32(1.0)
        else:
            weights[mode, y, x] = alpha
            for i in range(nmodes - 1):
                weights[i, y, x] *= alpha1

        for c in range(C):
            means[mode, c, y, x] = frame[c, y, x]
        vars_[mode, y, x] = var_init

        i = nmodes - 1
        while i > 0:
            if alpha < weights[i - 1, y, x]:
                break
            tw = weights[i, y, x];      weights[i, y, x]   = weights[i-1, y, x]; weights[i-1, y, x] = tw
            tv = vars_[i, y, x];        vars_[i, y, x]     = vars_[i-1, y, x];   vars_[i-1, y, x]   = tv
            for c in range(C):
                tm = means[i, c, y, x]; means[i, c, y, x] = means[i-1, c, y, x]; means[i-1, c, y, x] = tm
            i -= 1

    modes[y, x] = nmodes

    # ── classify ──────────────────────────────────────────────────────────────
    if background:
        mask[y, x] = uint8(0)
    elif detect_shadows and _detect_shadow(
            frame, y, x, C, nmodes, weights, means, vars_, Tb, TB, tau):
        mask[y, x] = shadow_val
    else:
        mask[y, x] = uint8(255)

    # ── update per-pixel alpha (SuBSENSE feedback) ────────────────────────────
    if adaptive_alpha:
        alpha_lo = alpha_base * float32(0.125)   # floor: base/8
        if background:
            new_alpha = alpha * _ALPHA_RESTORE
            if new_alpha > alpha_base:
                new_alpha = alpha_base
        else:
            new_alpha = alpha * _ALPHA_DECAY
            if new_alpha < alpha_lo:
                new_alpha = alpha_lo
        alpha_map[y, x] = new_alpha


# ── Kernel 2: spatial Gaussian propagation ────────────────────────────────────

@cuda.jit
def propagate_kernel(weights, means, vars_, modes, mask, alpha_map):
    """Copy the top Gaussian of each background pixel into a random neighbour.

    Uses a deterministic per-pixel hash as the random source so the kernel is
    reproducible and avoids cuRAND state overhead.  Every PHI-th background
    pixel (selected by frame-counter-free hash) propagates; on average 1/PHI
    of all background pixels update a neighbour per frame — matching ViBe's
    default phi=16.

    The target neighbour slot is always the lowest-weight (last) active
    component, so the propagated component must earn its place in the
    recipient's weight-sorted mixture before it dominates.
    """
    x, y = cuda.grid(2)
    H = weights.shape[2]   # weights shape: (K, H, W)  — wait, (K, H, W) no
    # weights is (K, H, W): first dim K, second H, third W
    # Recover H, W from the weight array dims
    H = weights.shape[1]
    W = weights.shape[2]
    if y >= H or x >= W:
        return

    # Only propagate from background pixels
    if mask[y, x] != uint8(0):
        return

    # Cheap deterministic hash — selects ~1/PHI pixels per call
    h = int32(y * 1000003 + x * 999983)
    if (h & int32(0x7FFFFFFF)) % _PHI != int32(0):
        return

    # Pick a neighbour index from the same hash
    nbr_idx = int32((h >> 4) & int32(7))   # 0..7
    ny = y + _NBR_DY[nbr_idx]
    nx = x + _NBR_DX[nbr_idx]
    if ny < 0 or ny >= H or nx < 0 or nx >= W:
        return

    K  = weights.shape[0]
    C  = means.shape[1]
    nm = int32(modes[y, x])
    if nm == int32(0):
        return

    # Source: component 0 (highest weight in the sorted mixture)
    src_w = weights[0, y, x]
    src_v = vars_[0, y, x]

    nm_nbr = int32(modes[ny, nx])

    if nm_nbr < K:
        # Append at the end (lowest-weight slot) and let the next mog2_step
        # insertion-sort it into its rightful place.
        slot = nm_nbr
        modes[ny, nx] = uint8(nm_nbr + 1)
    else:
        # Replace the lowest-weight existing component
        slot = K - 1

    # Write with a small seed weight so it doesn't immediately dominate
    weights[slot, ny, nx] = src_w * float32(0.1)
    vars_[slot, ny, nx]   = src_v
    for c in range(C):
        means[slot, c, ny, nx] = means[0, c, y, x]
