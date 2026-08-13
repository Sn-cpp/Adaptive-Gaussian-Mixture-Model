"""Separable background blur and mask morphology, parallelised with Numba.

`utils.post_processing` does the same job with OpenCV calls; this module is the
hand-written version that the CUDA kernels in `utils.blur_cuda` mirror, so the
CPU and GPU pipelines can be compared pixel for pixel.

A 15x15 Gaussian is 225 taps per pixel. The kernel is separable, so two 1D
passes cost 30 taps instead — and the vertical pass is fused with the composite,
which removes a full-frame write plus a full-frame read.
"""
import numpy as np
from numba import njit, prange

_JIT = dict(cache=True, fastmath=False)


@njit(parallel=True, **_JIT)
def blur_h(frame_bgr, tmp, k1d):
    """Horizontal pass: uint8 (H, W, 3) -> float32 tmp."""
    H = frame_bgr.shape[0]
    W = frame_bgr.shape[1]
    C = frame_bgr.shape[2]
    half = k1d.shape[0] // 2
    for y in prange(H):
        for x in range(W):
            for c in range(C):
                acc = np.float32(0.0)
                for t in range(-half, half + 1):
                    nx = min(max(x + t, 0), W - 1)
                    acc += frame_bgr[y, nx, c] * k1d[t + half]
                tmp[y, x, c] = acc
    return tmp


@njit(parallel=True, **_JIT)
def blur_v_composite(tmp, frame_bgr, mask, out, k1d):
    """Vertical pass fused with the composite.

    Foreground (mask == 255) keeps the sharp pixel; everything else — including
    shadows — takes the blurred value, so the blurred frame is never stored.
    """
    H = tmp.shape[0]
    W = tmp.shape[1]
    C = tmp.shape[2]
    half = k1d.shape[0] // 2
    for y in prange(H):
        for x in range(W):
            if mask[y, x] == 255:
                for c in range(C):
                    out[y, x, c] = frame_bgr[y, x, c]
            else:
                for c in range(C):
                    acc = np.float32(0.0)
                    for t in range(-half, half + 1):
                        ny = min(max(y + t, 0), H - 1)
                        acc += tmp[ny, x, c] * k1d[t + half]
                    v = int(acc + 0.5)
                    out[y, x, c] = min(max(v, 0), 255)
    return out


@njit(parallel=True, **_JIT)
def blur_2d(frame_bgr, out, kernel2d):
    """Non-separable baseline, kept only for the speedup comparison."""
    H = frame_bgr.shape[0]
    W = frame_bgr.shape[1]
    C = frame_bgr.shape[2]
    ks = kernel2d.shape[0]
    half = ks // 2
    for y in prange(H):
        for x in range(W):
            for c in range(C):
                acc = np.float32(0.0)
                for ky in range(ks):
                    ny = min(max(y + ky - half, 0), H - 1)
                    for kx in range(ks):
                        nx = min(max(x + kx - half, 0), W - 1)
                        acc += frame_bgr[ny, nx, c] * kernel2d[ky, kx]
                out[y, x, c] = min(max(int(acc + 0.5), 0), 255)
    return out


@njit(parallel=True, **_JIT)
def morph_close(mask, tmp_mask, out_mask):
    """3x3 dilate then erode on the foreground (mask == 255).

    CLOSE, not OPEN. The pipelines used to erode first, which is the operation
    measured on CDnet highway as the one that empties masks: `median + OPEN`
    scores F1 0.9182 and produces all 6 of the entirely-empty frames the old
    post-processing chain was blamed for, while a CLOSE of the same size is the
    second-best refinement measured (see `utils.post_processing.mask_refiner`).
    An erode-first pass deletes any structure thinner than the kernel outright,
    and a person's arm at low resolution is exactly that.

    Same two kernels, same cost, opposite order.
    """
    H = mask.shape[0]
    W = mask.shape[1]
    for y in prange(H):
        for x in range(W):
            v = np.uint8(0)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny = min(max(y + dy, 0), H - 1)
                    nx = min(max(x + dx, 0), W - 1)
                    if mask[ny, nx] == 255:
                        v = np.uint8(255)
            tmp_mask[y, x] = v
    for y in prange(H):
        for x in range(W):
            v = np.uint8(255)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny = min(max(y + dy, 0), H - 1)
                    nx = min(max(x + dx, 0), W - 1)
                    if tmp_mask[ny, nx] != 255:
                        v = np.uint8(0)
            out_mask[y, x] = v
    return out_mask


def warmup():
    bgr = np.zeros((4, 4, 3), np.uint8)
    tmp = np.zeros((4, 4, 3), np.float32)
    out = np.zeros((4, 4, 3), np.uint8)
    mask = np.zeros((4, 4), np.uint8)
    k1d = np.ones(3, np.float32) / 3
    blur_h(bgr, tmp, k1d)
    blur_v_composite(tmp, bgr, mask, out, k1d)
    blur_2d(bgr, out, np.ones((3, 3), np.float32) / 9)
    morph_close(mask, mask.copy(), mask.copy())
