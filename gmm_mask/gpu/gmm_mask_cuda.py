import numpy as np
from numba import cuda, float32, int32, uint8

from gmm_mask.gmm_mask_common import GMM_Mask_Base
from utils import to_planar

from settings import FLT_EPSILON

TILE_X = 32
TILE_Y = 8
MAX_C = 3

class GMM_Mask_CUDA(GMM_Mask_Base):
    def __init__(self, height: int, width: int, *args, **kwargs):
        super().__init__(height, width, *args, **kwargs)
        
        self.d_means   = cuda.to_device(self.means)
        self.d_vars    = cuda.to_device(self.vars)
        self.d_weights = cuda.to_device(self.weights)
        self.d_modes   = cuda.to_device(self.modes)
        self.d_mask    = cuda.to_device(self.mask)
        self.d_bg_prob = cuda.to_device(self.bg_prob)

        self.block = (TILE_X, TILE_Y)
        self.grid  = ((self.W  + TILE_X - 1) // TILE_X,
                        (self.H + TILE_Y - 1) // TILE_Y)

    def step_device(self, d_frame, args, stream=0):
        """Enqueue one MOG2 pass; returns (d_mask, d_bg_prob) without sync."""
        mog2_step_kernel[self.grid, self.block, stream](
            d_frame, self.d_weights, self.d_means, self.d_vars,
            self.d_modes, self.d_mask, self.d_bg_prob, *args)
        return self.d_mask, self.d_bg_prob

    def _step_kernel(self, frame, to_host, args):
        d_frame = cuda.to_device(frame)
        self.step_device(d_frame, args)
        cuda.synchronize()

        if to_host:
            self.d_mask.copy_to_host(self.mask)
            self.d_bg_prob.copy_to_host(self.bg_prob)
            return self.mask, self.bg_prob
        else:
            return self.d_mask, self.d_bg_prob

    def sync_state(self):
        """Copy the device model back to the host arrays (for tests)."""
        self.d_weights.copy_to_host(self.weights)
        self.d_means.copy_to_host(self.means)
        self.d_vars.copy_to_host(self.vars)
        self.d_modes.copy_to_host(self.modes)

    def background_image(self):
        self.sync_state()
        return super().background_image()

def is_available():
    try:
        return cuda.is_available()
    except Exception:
        return False


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
def mog2_step_kernel(frame, weights, means, vars_, modes, mask, bg_prob,
                     alpha, prune, Tb, Tg, TB, var_init, var_min, var_max,
                     tau, shadow_val, detect_shadows):
    """One MOG2 step: one thread per pixel.

    Writes classification result into `mask` and background probability into
    `bg_prob` (sum of weights of Gaussian components with dist2 < Tb*var).
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
