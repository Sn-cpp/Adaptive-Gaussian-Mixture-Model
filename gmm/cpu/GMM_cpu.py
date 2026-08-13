"""Sequential MOG2 reference — plain Python, one pixel at a time.

This file is the *specification*. `GMM_CPU_NUMBA` and `GMM_CUDA` are
line-by-line transliterations of `mog2_step` below; if they ever disagree, this
one is right.
"""
import numpy as np

from settings import FLT_EPSILON
from gmm.mog2_common import MOG2Base


def detect_shadow(frame, y, x, C, nmodes, weights, means, vars_, Tb, TB, tau):
    """Port of OpenCV `detectShadowGMM`."""
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


def mog2_step(frame, weights, means, vars_, modes, mask, bg_prob,
              alpha_in, prune_in, Tb, Tg, TB, var_init, var_min, var_max,
              tau, shadow_val, detect_shadows, conservative, Te):
    K = weights.shape[0]
    C = means.shape[1]
    H, W = frame.shape[1], frame.shape[2]
    alpha1_in = np.float32(1.0) - alpha_in
    dData = [np.float32(0.0)] * C

    for y in range(H):
        for x in range(W):
            # Conservative update: `mask` still holds the previous frame's
            # decision at this point, and a pixel that was foreground then does
            # not get to teach the background model anything now. alpha = 0
            # leaves the means and variances alone and stops a new mode being
            # created; prune = 0 is just as necessary, or the weights would
            # keep decaying and the modes would be pruned away underneath us.
            #
            # Held only while the pixel is still *far* from the background it
            # was frozen at, measured against the dominant mode with the loose
            # threshold Te. Without that gate a global appearance change
            # freezes every pixel at the old exposure and none can ever track
            # the new one — see MOG2_PROTECT_EXIT.
            alpha, prune, alpha1 = alpha_in, prune_in, alpha1_in
            if conservative and mask[y, x] == 255 and modes[y, x] > 0:
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
            nmodes = int(modes[y, x])
            total_weight = np.float32(0.0)
            bg_weight_sum = np.float32(0.0)

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
                        bg_weight_sum += weights[mode, y, x]

                    if dist2 < Tg * var:
                        fits_pdf = True
                        weight += alpha
                        k = alpha / weight            # k from the *new* weight

                        for c in range(C):
                            means[mode, c, y, x] -= k * dData[c]

                        varnew = var + k * (dist2 - var)
                        varnew = max(varnew, var_min)
                        varnew = min(varnew, var_max)
                        vars_[mode, y, x] = varnew

                        # bubble the matched mode up while it outweighs its neighbour
                        i = mode
                        while i > 0:
                            if weight < weights[i - 1, y, x]:
                                break
                            swap_count += 1
                            weights[i, y, x], weights[i - 1, y, x] = \
                                weights[i - 1, y, x], weights[i, y, x]
                            vars_[i, y, x], vars_[i - 1, y, x] = \
                                vars_[i - 1, y, x], vars_[i, y, x]
                            for c in range(C):
                                means[i, c, y, x], means[i - 1, c, y, x] = \
                                    means[i - 1, c, y, x], means[i, c, y, x]
                            i -= 1

                if weight < -prune:      # complexity reduction: drop the mode
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
                    mode = K - 1                      # replace the weakest
                else:
                    mode = nmodes                     # append a new one
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
                    weights[i, y, x], weights[i - 1, y, x] = \
                        weights[i - 1, y, x], weights[i, y, x]
                    vars_[i, y, x], vars_[i - 1, y, x] = \
                        vars_[i - 1, y, x], vars_[i, y, x]
                    for c in range(C):
                        means[i, c, y, x], means[i - 1, c, y, x] = \
                            means[i - 1, c, y, x], means[i, c, y, x]
                    i -= 1

            modes[y, x] = nmodes

            bg_prob[y, x] = bg_weight_sum

            if background:
                mask[y, x] = 0
            elif detect_shadows and detect_shadow(
                    frame, y, x, C, nmodes, weights, means, vars_, Tb, TB, tau):
                mask[y, x] = shadow_val
            else:
                mask[y, x] = 255

    return mask


class GMM_CPU(MOG2Base):
    """Sequential MOG2. Same constructor contract as `GMM_CPU`."""
    FILLS_BG_PROB = True

    def _step_kernel(self, frame, args):
        mog2_step(frame, self.weights, self.means, self.vars,
                  self.modes, self.mask, self.bg_prob, *args)
