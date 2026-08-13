"""Numba MOG2 — same algorithm as `GMM_cpu`, `prange` over rows.

Each row owns its pixels' state, so there is no synchronisation at all.
"""
import numpy as np
from numba import njit, prange

from settings import FLT_EPSILON
from gmm.mog2_common import MOG2Base

_JIT = dict(cache=True, fastmath=False)


@njit(cache=True)
def _detect_shadow(frame, y, x, C, nmodes, weights, means, vars_, Tb, TB, tau):
    t_weight = np.float32(0.0)
    for mode in range(nmodes):
        num = np.float32(0.0)
        den = np.float32(0.0)
        for c in range(C):
            num += frame[c, y, x] * means[mode, c, y, x]
            den += means[mode, c, y, x] * means[mode, c, y, x]
        if den == 0.0:
            return False
        if num <= den and num >= tau * den:
            a = num / den
            dist2a = np.float32(0.0)
            for c in range(C):
                dD = a * means[mode, c, y, x] - frame[c, y, x]
                dist2a += dD * dD
            if dist2a < Tb * vars_[mode, y, x] * a * a:
                return True
        t_weight += weights[mode, y, x]
        if t_weight > TB:
            return False
    return False


@njit(parallel=True, **_JIT)
def mog2_step(frame, weights, means, vars_, modes, mask, bg_prob,
              alpha_in, prune_in, Tb, Tg, TB, var_init, var_min, var_max,
              tau, shadow_val, detect_shadows, conservative, Te):
    K = weights.shape[0]
    C = means.shape[1]
    H = frame.shape[1]
    W = frame.shape[2]
    alpha1_in = np.float32(1.0) - alpha_in

    for y in prange(H):
        dData = np.empty(C, dtype=np.float32)
        for x in range(W):
            # Conservative update — see `GMM_cpu.mog2_step`, same lines.
            # `mask` still holds the previous frame's decision, and Te is the
            # loose exit threshold that stops a global appearance change
            # latching the whole frame.
            alpha = alpha_in
            prune = prune_in
            alpha1 = alpha1_in
            if conservative and mask[y, x] == np.uint8(255) and modes[y, x] > 0:
                d0 = np.float32(0.0)
                for c in range(C):
                    dd = means[0, c, y, x] - frame[c, y, x]
                    d0 += dd * dd
                if d0 >= Te * vars_[0, y, x]:
                    alpha = np.float32(0.0)
                    prune = np.float32(0.0)
                    alpha1 = np.float32(1.0)

            background = False
            fits_pdf = False
            nmodes = np.int32(modes[y, x])
            total_weight = np.float32(0.0)
            bg_weight_sum = np.float32(0.0)

            mode = 0
            while mode < nmodes:
                weight = alpha1 * weights[mode, y, x] + prune
                swap_count = 0

                if not fits_pdf:
                    var = vars_[mode, y, x]
                    dist2 = np.float32(0.0)

                    # Weighting the channels unequally here — down-weighting
                    # luma to 0.25 so shadow counts for less — was tried and
                    # measured on CDnet highway (400 frames, YCrCb input):
                    #
                    #   (Y,Cr,Cb) = 1,1,1      F1 0.9297   P 0.976  R 0.887
                    #   (Y,Cr,Cb) = 0.5,1,1    F1 0.9081   P 0.995  R 0.835
                    #   (Y,Cr,Cb) = 0.25,1,1   F1 0.8663   P 0.999  R 0.765
                    #   (Y,Cr,Cb) = 0.1,1,1    F1 0.8115   P 1.000  R 0.683
                    #
                    # It buys precision and pays far more in recall: a vehicle
                    # differs from asphalt mostly in brightness, so discounting
                    # luma stops detecting it. Feeding the model YCrCb already
                    # captures the shadow benefit (F1 0.73 -> 0.86); weighting
                    # on top of that overshoots. Left unweighted deliberately.

                    for c in range(C):
                        dd = means[mode, c, y, x] - frame[c, y, x]
                        dData[c] = dd
                        dist2 += dd * dd

                    if total_weight < TB and dist2 < Tb * var:
                        background = True
                        bg_weight_sum += weights[mode, y, x]

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
                    weight = np.float32(0.0)
                    nmodes -= 1

                weights[mode - swap_count, y, x] = weight
                total_weight += weight
                mode += 1

            inv_weight = np.float32(0.0)
            if abs(total_weight) > FLT_EPSILON:
                inv_weight = np.float32(1.0) / total_weight
            for mode in range(nmodes):
                weights[mode, y, x] *= inv_weight

            if not fits_pdf and alpha > 0.0:
                if nmodes == K:
                    mode = K - 1
                else:
                    mode = nmodes
                    nmodes += 1

                if nmodes == 1:
                    weights[mode, y, x] = np.float32(1.0)
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

            bg_prob[y, x] = bg_weight_sum

            if background:
                mask[y, x] = 0
            elif detect_shadows and _detect_shadow(
                    frame, y, x, C, nmodes, weights, means, vars_, Tb, TB, tau):
                mask[y, x] = shadow_val
            else:
                mask[y, x] = 255

    return mask


def warmup(C=3, K=5):
    """JIT-compile on tiny arrays so benchmarks exclude compilation."""
    from gmm.mog2_common import kernel_args
    frame = np.zeros((C, 4, 4), np.float32)
    mog2_step(frame, np.zeros((K, 4, 4), np.float32),
              np.zeros((K, C, 4, 4), np.float32), np.zeros((K, 4, 4), np.float32),
              np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8),
              np.zeros((4, 4), np.float32),
              *kernel_args(0.01))


class GMM_CPU_NUMBA(MOG2Base):
    """Numba-parallel MOG2. Same constructor contract as `GMM_CPU_NUMBA`."""
    FILLS_BG_PROB = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warmup(self.n_channels, self.n_comps)

    def _step_kernel(self, frame, args):
        mog2_step(frame, self.weights, self.means, self.vars,
                  self.modes, self.mask, self.bg_prob, *args)
