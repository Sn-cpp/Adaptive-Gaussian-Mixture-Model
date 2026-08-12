"""Shared helpers for the MOG2 models.

The per-pixel algorithm implemented by `GMM_CPU`, `GMM_CPU_NUMBA` and
`GMM_CUDA` is a direct port of OpenCV's `BackgroundSubtractorMOG2`
(Zivkovic 2004, `bgfg_gaussmix2.cpp`). State is kept in the same planar layout
the rest of the project uses:

    means   (K, C, H, W)
    vars    (K, H, W)
    weights (K, H, W)
    modes   (H, W)  uint8 — number of *active* Gaussians for that pixel

Frames are planar too, `(C, H, W)` float32, exactly like `GMM_CPU`.
"""
from time import perf_counter

import cv2
import numpy as np

from settings import (
    MOG2_BACKGROUND_RATIO, MOG2_CT, MOG2_DETECT_SHADOWS, MOG2_HISTORY,
    MOG2_SHADOW_TAU, MOG2_SHADOW_VALUE, MOG2_VAR_INIT, MOG2_VAR_MAX,
    MOG2_VAR_MIN, MOG2_VAR_THRESHOLD, MOG2_VAR_THRESHOLD_GEN,
    BLUR_KSIZE, BLUR_SIGMA,
)


def learning_rate(rate=-1.0, nframes=1, history=MOG2_HISTORY):
    """OpenCV semantics.

    A negative rate means "auto": OpenCV ramps alpha as 1/min(2*nframes,
    history) so the model converges quickly on the first frames and then
    settles at 1/history. `nframes` is 1-based.
    """
    if rate >= 0 and nframes > 1:
        return float(rate)
    return 1.0 / min(2 * nframes, history)


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


def to_planar(frame_bgr, color=True):
    """(H, W, 3) uint8 BGR -> (C, H, W) float32 model input."""
    if color:
        return np.ascontiguousarray(frame_bgr.transpose(2, 0, 1).astype(np.float32))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(gray.astype(np.float32)[None])


def create_gaussian_kernel_1d(ksize=BLUR_KSIZE, sigma=BLUR_SIGMA):
    """Separable Gaussian: one 1D pass per axis instead of one 2D pass."""
    ax = np.arange(ksize, dtype=np.float32) - ksize // 2
    k = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def create_gaussian_kernel(ksize=BLUR_KSIZE, sigma=BLUR_SIGMA):
    """2D kernel — kept as the non-separable baseline for the benchmarks."""
    k1 = create_gaussian_kernel_1d(ksize, sigma)
    return np.outer(k1, k1).astype(np.float32)


def background_image(means, vars_, weights, modes,
                     background_ratio=MOG2_BACKGROUND_RATIO):
    """Port of OpenCV `getBackgroundImage`: weighted mean of the background modes.

    Returns (H, W, C) uint8.
    """
    k_max, C, H, W = means.shape
    total = np.zeros((H, W), np.float32)
    acc = np.zeros((C, H, W), np.float32)
    for k in range(k_max):
        use = (k < modes) & (total <= background_ratio)
        wk = np.where(use, weights[k], 0.0).astype(np.float32)
        acc += wk[None] * means[k]
        total += wk
    total = np.where(total > 0, total, 1.0)
    return np.clip((acc / total[None]).transpose(1, 2, 0) + 0.5, 0, 255).astype(np.uint8)


def opencv_reference(history=MOG2_HISTORY, var_threshold=MOG2_VAR_THRESHOLD,
                     detect_shadows=MOG2_DETECT_SHADOWS):
    """The ground truth every correctness test is measured against."""
    return cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=float(var_threshold),
        detectShadows=detect_shadows,
    )


class MOG2Base:
    #: Does this backend fill `bg_prob`? A backend that leaves it at zero is
    #: not broken — the mask is all most callers want — but anything reading
    #: the confidence must refuse to run rather than segment a field of zeros.
    FILLS_BG_PROB = False

    """State allocation and parameter bookkeeping shared by the three backends.

    Subclasses only implement `_step_kernel`. Nothing is ever reallocated after
    construction.
    """

    def __init__(self, first_frame, n_components, color=True, history=MOG2_HISTORY,
                 var_threshold=MOG2_VAR_THRESHOLD,
                 var_threshold_gen=MOG2_VAR_THRESHOLD_GEN,
                 background_ratio=MOG2_BACKGROUND_RATIO,
                 var_init=MOG2_VAR_INIT, var_min=MOG2_VAR_MIN,
                 var_max=MOG2_VAR_MAX, ct=MOG2_CT, tau=MOG2_SHADOW_TAU,
                 shadow_value=MOG2_SHADOW_VALUE,
                 detect_shadows=MOG2_DETECT_SHADOWS, *arg, **kwargs):
        self.height, self.width = first_frame.shape[:2]
        self.n_comps = int(n_components)
        self.n_channels = 3 if color else 1
        self.color = color
        self.history = history
        self.nframes = 0
        self._params = dict(
            var_threshold=var_threshold, var_threshold_gen=var_threshold_gen,
            background_ratio=background_ratio, var_init=var_init,
            var_min=var_min, var_max=var_max, ct=ct, tau=tau,
            shadow_value=shadow_value, detect_shadows=detect_shadows,
        )

        K, C, H, W = self.n_comps, self.n_channels, self.height, self.width
        # MOG2 starts with zero active modes — the first frame creates mode 0.
        self.means = np.zeros((K, C, H, W), dtype=np.float32)
        self.vars = np.zeros((K, H, W), dtype=np.float32)
        self.weights = np.zeros((K, H, W), dtype=np.float32)
        self.modes = np.zeros((H, W), dtype=np.uint8)
        self.mask = np.zeros((H, W), dtype=np.uint8)
        # Background confidence in [0, 1]: the weight of every mode that
        # matched this pixel inside the background set. Classification
        # already computes it and used to discard it; keeping it costs one
        # store and hands the graph cut a data term for free.
        self.bg_prob = np.zeros((H, W), dtype=np.float32)

    def next_args(self, update_alpha=-1.0):
        """Per-frame scalar args — alpha follows OpenCV's warm-up ramp."""
        self.nframes += 1
        self.alpha = learning_rate(update_alpha, self.nframes, self.history)
        return kernel_args(self.alpha, **self._params)

    def step(self, frame, update_alpha=-1.0):
        """One MOG2 pass over a planar (C, H, W) float32 frame -> (mask, seconds).

        Same signature and return shape as `GMM_CPU.step` and friends, so every
        model in `gmm` can be driven identically by `main.py` and `benchmark.py`.

        MOG2 does classification and update in a single fused traversal, so
        there is no separate `predict` / `update` pair here.

        A negative `update_alpha` selects OpenCV's ramp `1/min(2*nframes, history)`.
        """
        
        t0 = perf_counter()
        self._step_kernel(frame, self.next_args(update_alpha))
        return self.mask, perf_counter() - t0

    def background_image(self):
        return background_image(self.means, self.vars, self.weights, self.modes,
                                self._params['background_ratio'])

    def _step_kernel(self, frame, args):     # pragma: no cover - interface
        raise NotImplementedError
