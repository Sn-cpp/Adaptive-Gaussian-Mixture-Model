"""v2 — fuse what can be fused, tile what cannot.

v1 runs three kernels per frame: MOG2 step, threshold, median. v2 runs two.

**What fuses.** The threshold reads `bg_prob` at pixel (y, x) and writes one
byte at (y, x) — no neighbours, no other pixel's result. The MOG2 kernel has
`bg_weight_sum` sitting in a register at exactly that point. So the threshold
is not a kernel at all in v2; it is a single assignment in the model
kernel's epilogue,
and it removes a full-frame write followed by a full-frame read.

**What does not.** The median needs its neighbours' *post-threshold* values,
and a CUDA kernel has no grid-wide barrier — a neighbouring block may not have
run yet. Fusing it would be a correctness bug that mostly does not show up in
testing, which is the worst kind. It stays a separate kernel, and is optimised
instead: the tiled version loads each pixel once per block rather than 25
times.

Measured effect is in `bench_post.py`. The numbers to expect: the fusion saves
one kernel launch and 2 x H x W bytes of traffic; the tiling turns 25 global
loads per pixel into about 1.4.
"""
import numpy as np
from numba import cuda, float32, int32, uint8

from gmm_mask.gpu import blur_kernels as bk
from gmm_mask.gpu import post_kernels as pk
from gmm_mask.gpu.gmm_mask_cuda_v1 import (
    GMM_Mask_CUDA_v1, TILE_X, TILE_Y, MAX_C, _detect_shadow,
)
from settings import FLT_EPSILON


class GMM_Mask_CUDA_v2(GMM_Mask_CUDA_v1):
    """Same output as v1, two kernels instead of three.

    Inherits v1's buffers and flags; only the enqueue order changes. The
    refined mask is verified against v1 pixel-for-pixel in `tests/`.
    """

    def _blur_kernels(self):
        """The tiled pair. Same output as v1's, each pixel read once per block.

        Overriding one method is the whole difference, which is the point: if
        the tiled kernels turn out to be no faster than the naive ones on a T4
        — plausible, since L2 already serves row-strided reuse well — that is a
        measurement to report, not a rewrite to undo.
        """
        return bk.blur_h_tiled_kernel, bk.blur_v_composite_tiled_kernel

    def step_device(self, d_frame, args, stream=0):
        if not self.post:
            return super().step_device(d_frame, args, stream)

        mog2_step_fused_kernel[self.grid, self.block, stream](
            d_frame,
            self.d_weights, self.d_means, self.d_vars,
            self.d_modes, self.d_mask, self.d_bg_prob,
            self.d_post_a,                 # binary mask, written in the epilogue
            self.bg_prob_threshold,
            *args,
        )
        pk.median5_tiled_kernel[self.grid, self.block, stream](
            self.d_post_a, self.d_post_b)
        return self.d_post_b, self.d_bg_prob


@cuda.jit
def mog2_step_fused_kernel(frame, weights, means, vars_, modes, mask, bg_prob,
                           refined, bg_thresh,
                           alpha, prune, Tb, Tg, TB,
                           var_init, var_min, var_max,
                           tau, shadow_val, detect_shadows):
    """MOG2 step with the confidence threshold folded into the epilogue.

    A transliteration of `mog2_step_v1_kernel` with `adaptive_alpha` off — the
    adaptive path is not carried into v2 because it is unmeasured on the car
    dataset and fusing an unvalidated variant would make the parity test
    meaningless. The only additions are the final assignment.
    """
    x, y = cuda.grid(2)
    H = frame.shape[1]
    W = frame.shape[2]
    if y >= H or x >= W:
        return

    K = weights.shape[0]
    C = means.shape[1]
    alpha1 = float32(1.0) - alpha
    dData = cuda.local.array(MAX_C, float32)

    background = False
    fits_pdf = False
    nmodes = int32(modes[y, x])
    total_weight = float32(0.0)
    bg_weight_sum = float32(0.0)

    mode = 0
    while mode < nmodes:
        weight = alpha1 * weights[mode, y, x] + prune
        swap_count = 0

        if not fits_pdf:
            var = vars_[mode, y, x]
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
                weight += alpha
                k = alpha / weight
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
                    tw = weights[i, y, x]
                    weights[i, y, x] = weights[i - 1, y, x]
                    weights[i - 1, y, x] = tw
                    tv = vars_[i, y, x]
                    vars_[i, y, x] = vars_[i - 1, y, x]
                    vars_[i - 1, y, x] = tv
                    for c in range(C):
                        tm = means[i, c, y, x]
                        means[i, c, y, x] = means[i - 1, c, y, x]
                        means[i - 1, c, y, x] = tm
                    i -= 1

        if weight < -prune:
            weight = float32(0.0)
            nmodes -= 1

        weights[mode - swap_count, y, x] = weight
        total_weight += weight
        mode += 1

    inv_weight = float32(0.0)
    if abs(total_weight) > FLT_EPSILON:
        inv_weight = float32(1.0) / total_weight
    for mode in range(nmodes):
        weights[mode, y, x] *= inv_weight

    bg_prob[y, x] = bg_weight_sum

    if not fits_pdf and alpha > float32(0.0):
        if nmodes == K:
            mode = K - 1
        else:
            mode = nmodes
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
            tw = weights[i, y, x]
            weights[i, y, x] = weights[i - 1, y, x]
            weights[i - 1, y, x] = tw
            tv = vars_[i, y, x]
            vars_[i, y, x] = vars_[i - 1, y, x]
            vars_[i - 1, y, x] = tv
            for c in range(C):
                tm = means[i, c, y, x]
                means[i, c, y, x] = means[i - 1, c, y, x]
                means[i - 1, c, y, x] = tm
            i -= 1

    modes[y, x] = nmodes

    if background:
        mask[y, x] = uint8(0)
    elif detect_shadows and _detect_shadow(
            frame, y, x, C, nmodes, weights, means, vars_, Tb, TB, tau):
        mask[y, x] = shadow_val
    else:
        mask[y, x] = uint8(255)

    # ── fused epilogue: the threshold, without a second kernel ────────────────
    refined[y, x] = uint8(255) if bg_weight_sum < bg_thresh else uint8(0)
