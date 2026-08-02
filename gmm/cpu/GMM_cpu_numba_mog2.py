"""Numba MOG2 — same algorithm as `GMM_cpu_mog2`, `prange` over rows.

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
def mog2_step(frame, weights, means, vars_, modes, mask,
              alpha, prune, Tb, Tg, TB, var_init, var_min, var_max,
              tau, shadow_val, detect_shadows):
    K = weights.shape[0]
    C = means.shape[1]
    H = frame.shape[1]
    W = frame.shape[2]
    alpha1 = np.float32(1.0) - alpha

    for y in prange(H):
        dData = np.empty(C, dtype=np.float32)
        for x in range(W):
            background = False
            fits_pdf = False
            nmodes = np.int32(modes[y, x])
            total_weight = np.float32(0.0)

            mode = 0
            while mode < nmodes:
                weight = alpha1 * weights[mode, y, x] + prune
                swap_count = 0

                if not fits_pdf:
                    var = vars_[mode, y, x]
                    dist2 = np.float32(0.0)
                    for c in range(C):
                        dd = means[mode, c, y, x] - frame[c, y, x]
                        dData[c] = dd
                        dist2 += dd * dd

                    if total_weight < TB and dist2 < Tb * var:
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
              *kernel_args(0.01))


class GMM_CPU_NUMBA_MOG2(MOG2Base):
    """Numba-parallel MOG2. Same constructor contract as `GMM_CPU_NUMBA`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warmup(self.n_channels, self.n_comps)

    def _step_kernel(self, frame, args):
        mog2_step(frame, self.weights, self.means, self.vars,
                  self.modes, self.mask, *args)
