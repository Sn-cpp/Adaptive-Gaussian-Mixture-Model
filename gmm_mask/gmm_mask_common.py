import numpy as np
from time import perf_counter

from utils import to_planar

from settings import *

class GMM_Mask_Base:
    def __init__(self, height, width, *args, **kwargs):
        
        self.K = np.int32(MOG2_N_COMPONENTS)
        self.C = np.int32(3 if MOG2_COLOR else 1)
        self.H = np.int32(height)
        self.W = np.int32(width)
        
        self.color = MOG2_COLOR
        self.history = MOG2_HISTORY
        self.nframes = 0
        self._params = dict(
            var_threshold=MOG2_VAR_THRESHOLD,
            var_threshold_gen=MOG2_VAR_THRESHOLD_GEN,
            background_ratio=MOG2_BACKGROUND_RATIO,
            var_init=MOG2_VAR_INIT, var_min=MOG2_VAR_MIN,
            var_max=MOG2_VAR_MAX, ct=MOG2_CT, tau=MOG2_SHADOW_TAU,
            shadow_value=MOG2_SHADOW_VALUE,
            detect_shadows=MOG2_DETECT_SHADOWS,
        )

        K, C, H, W = self.K, self.C, self.H, self.W
        # MOG2 starts with zero active modes — the first frame creates mode 0.
        self.means = np.zeros((K, C, H, W), dtype=np.float32)
        self.vars = np.zeros((K, H, W), dtype=np.float32)
        self.weights = np.zeros((K, H, W), dtype=np.float32)
        self.modes = np.zeros((H, W), dtype=np.uint8)
        self.mask = np.zeros((H, W), dtype=np.uint8)
        self.bg_prob = np.zeros((H, W), dtype=np.float32)

    def apply(self, frame: np.ndarray, update_alpha=-1.0, to_host=True):
        frame_planar = to_planar(frame)

        t0 = perf_counter()
        mask, bg_score = self._step_kernel(frame_planar, to_host, self.next_args(update_alpha))
        return mask, bg_score, perf_counter() - t0

    def _step_kernel(self, frame: np.ndarray, to_host: bool, args):
        raise NotImplementedError

    def next_args(self, update_alpha):
        """Per-frame scalar args — alpha follows OpenCV's warm-up ramp."""
        self.nframes += 1
        self.alpha = self.learning_rate(update_alpha)
        return kernel_args(self.alpha, **self._params)

    def background_image(self):
        """The learned background as a BGR/YCrCb image, for the demo and report.

        Three subclasses already override this to sync device state first and
        then call `super()`, but no base implementation ever existed — every one
        of those overrides raised `AttributeError` on the first call. Nothing in
        the shipping pipeline calls it, which is why it went unnoticed; the
        notebook wants the picture, so the base method is the smaller fix.

        MOG2's background is not one Gaussian: it is the leading modes whose
        cumulative weight passes `TB`. Averaging their means weighted by weight
        is the same rule the mask uses, so the picture and the mask agree about
        what "background" means. Pixels with no active mode yet (the first
        frame) come back black rather than as a division by zero.
        """
        w = self.weights                       # (K, H, W), sorted descending
        # The kernel tests `total_weight < TB` *before* adding the current
        # mode's weight, so the reconstruction is the shifted cumulative sum,
        # not `cumsum - w`. The two agree on every one of 200 000 random
        # normalised weight vectors, but `(a + b) - b == a` is not an identity
        # in float32 and the shifted form is what the kernel actually holds in
        # its accumulator. Free to get right, so get it right.
        cum = np.cumsum(w, axis=0)
        before = np.concatenate(
            [np.zeros_like(w[:1]), cum[:-1]], axis=0)
        is_bg = before < np.float32(self._params["background_ratio"])
        is_bg &= np.arange(self.K)[:, None, None] < self.modes[None]

        wm = np.where(is_bg, w, np.float32(0.0))          # (K, H, W)
        denom = wm.sum(axis=0)                            # (H, W)
        acc = (wm[:, None] * self.means).sum(axis=0)      # (C, H, W)
        out = np.divide(acc, denom, out=np.zeros_like(acc), where=denom > 0)
        # Round, do not truncate: OpenCV's getBackgroundImage rounds, and a
        # learned mean of 0.51 is a 1, not a 0.
        return np.clip(np.rint(out), 0, 255).astype(np.uint8).transpose(1, 2, 0)

    def learning_rate(self, rate=-1.0):
        """OpenCV semantics.

        A negative rate means "auto": OpenCV ramps alpha as 1/min(2*nframes,
        history) so the model converges quickly on the first frames and then
        settles at 1/history. `nframes` is 1-based.
        """
        if rate >= 0 and self.nframes > 1:
            return float(rate)
        return 1.0 / min(2 * self.nframes, self.history)


def kernel_args(alpha, var_threshold=MOG2_VAR_THRESHOLD,
            var_threshold_gen=MOG2_VAR_THRESHOLD_GEN,
            background_ratio=MOG2_BACKGROUND_RATIO,
            var_init=MOG2_VAR_INIT, var_min=MOG2_VAR_MIN,
            var_max=MOG2_VAR_MAX, ct=MOG2_CT, tau=MOG2_SHADOW_TAU,
            shadow_value=MOG2_SHADOW_VALUE,
            detect_shadows=MOG2_DETECT_SHADOWS):
    """Scalar arguments every backend kernel takes, in one fixed order."""
    return (
        np.float32(alpha),
        np.float32(-alpha * ct),        # prune
        np.float32(var_threshold),      # Tb
        np.float32(var_threshold_gen),  # Tg
        np.float32(background_ratio),   # TB
        np.float32(var_init),
        np.float32(var_min),
        np.float32(var_max),
        np.float32(tau),
        np.uint8(shadow_value),
        bool(detect_shadows),
    )
