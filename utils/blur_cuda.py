"""CUDA counterparts of `utils.blur_numba`.

Both blur passes cache their input in shared memory with a halo, so each pixel
of the 15-tap window is read from global memory once per block instead of once
per thread. The vertical pass writes the composite directly.

`HALO` is baked into the shared-memory tile at compile time, so the kernels are
built for `settings.BLUR_KSIZE`; change that constant and reimport to use a
different radius.
"""
import numpy as np
from numba import cuda, float32, int32, uint8

from settings import BLUR_KSIZE

TILE_X = 32
TILE_Y = 8
HALO = BLUR_KSIZE // 2
SH_W = TILE_X + 2 * HALO
SH_H = TILE_Y + 2 * HALO
MAX_C = 3


@cuda.jit
def blur_h_kernel(frame_bgr, tmp, k1d):
    """Horizontal pass. Each block caches TILE_X + 2*HALO columns."""
    tile = cuda.shared.array((TILE_Y, SH_W, MAX_C), float32)
    H = frame_bgr.shape[0]
    W = frame_bgr.shape[1]
    C = frame_bgr.shape[2]

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    x = cuda.blockIdx.x * TILE_X + tx
    y = cuda.blockIdx.y * TILE_Y + ty
    yc = min(max(y, 0), H - 1)

    # every thread loads its own column plus a slice of the two halo strips
    for i in range(tx, SH_W, TILE_X):
        sx = min(max(cuda.blockIdx.x * TILE_X + i - HALO, 0), W - 1)
        for c in range(C):
            tile[ty, i, c] = float32(frame_bgr[yc, sx, c])
    cuda.syncthreads()

    if y < H and x < W:
        for c in range(C):
            acc = float32(0.0)
            for t in range(2 * HALO + 1):
                acc += tile[ty, tx + t, c] * k1d[t]
            tmp[y, x, c] = acc


@cuda.jit
def blur_v_composite_kernel(tmp, frame_bgr, mask, out, k1d):
    """Vertical pass fused with the composite — no blurred frame in global memory."""
    tile = cuda.shared.array((SH_H, TILE_X, MAX_C), float32)
    H = tmp.shape[0]
    W = tmp.shape[1]
    C = tmp.shape[2]

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    x = cuda.blockIdx.x * TILE_X + tx
    y = cuda.blockIdx.y * TILE_Y + ty
    xc = min(max(x, 0), W - 1)

    for i in range(ty, SH_H, TILE_Y):
        sy = min(max(cuda.blockIdx.y * TILE_Y + i - HALO, 0), H - 1)
        for c in range(C):
            tile[i, tx, c] = tmp[sy, xc, c]
    cuda.syncthreads()

    if y >= H or x >= W:
        return

    if mask[y, x] == uint8(255):
        for c in range(C):
            out[y, x, c] = frame_bgr[y, x, c]
    else:
        for c in range(C):
            acc = float32(0.0)
            for t in range(2 * HALO + 1):
                acc += tile[ty + t, tx, c] * k1d[t]
            v = int32(acc + float32(0.5))
            out[y, x, c] = uint8(min(max(v, int32(0)), int32(255)))


@cuda.jit
def blur_2d_kernel(frame_bgr, out, kernel2d):
    """Non-separable baseline, kept only for the speedup comparison."""
    x, y = cuda.grid(2)
    H = frame_bgr.shape[0]
    W = frame_bgr.shape[1]
    C = frame_bgr.shape[2]
    if y >= H or x >= W:
        return
    ks = kernel2d.shape[0]
    half = ks // 2
    for c in range(C):
        acc = float32(0.0)
        for ky in range(ks):
            ny = min(max(y + ky - half, 0), H - 1)
            for kx in range(ks):
                nx = min(max(x + kx - half, 0), W - 1)
                acc += float32(frame_bgr[ny, nx, c]) * kernel2d[ky, kx]
        out[y, x, c] = uint8(min(max(int32(acc + float32(0.5)), int32(0)), int32(255)))


@cuda.jit
def erode_kernel(mask, out):
    x, y = cuda.grid(2)
    H = mask.shape[0]
    W = mask.shape[1]
    if y >= H or x >= W:
        return
    v = uint8(255)
    for dy in range(-1, 2):
        ny = min(max(y + dy, 0), H - 1)
        for dx in range(-1, 2):
            nx = min(max(x + dx, 0), W - 1)
            if mask[ny, nx] != uint8(255):
                v = uint8(0)
    out[y, x] = v


@cuda.jit
def dilate_kernel(mask, out):
    x, y = cuda.grid(2)
    H = mask.shape[0]
    W = mask.shape[1]
    if y >= H or x >= W:
        return
    v = uint8(0)
    for dy in range(-1, 2):
        ny = min(max(y + dy, 0), H - 1)
        for dx in range(-1, 2):
            nx = min(max(x + dx, 0), W - 1)
            if mask[ny, nx] == uint8(255):
                v = uint8(255)
    out[y, x] = v


def blur_variants(frame_bgr, n=20, ksize=BLUR_KSIZE, sigma=None):
    """Time separable+tiled vs naive 2D with everything already on the device."""
    import time
    from gmm.mog2_common import create_gaussian_kernel, create_gaussian_kernel_1d
    from settings import BLUR_SIGMA

    sigma = BLUR_SIGMA if sigma is None else sigma
    H, W = frame_bgr.shape[:2]
    d_in = cuda.to_device(np.ascontiguousarray(frame_bgr))
    d_tmp = cuda.device_array((H, W, 3), np.float32)
    d_out = cuda.device_array((H, W, 3), np.uint8)
    d_mask = cuda.to_device(np.zeros((H, W), np.uint8))
    d_k1 = cuda.to_device(create_gaussian_kernel_1d(ksize, sigma))
    d_k2 = cuda.to_device(create_gaussian_kernel(ksize, sigma))
    block = (TILE_X, TILE_Y)
    grid = ((W + TILE_X - 1) // TILE_X, (H + TILE_Y - 1) // TILE_Y)

    def timed(fn):
        fn()
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        cuda.synchronize()
        return (time.perf_counter() - t0) / n

    sep = timed(lambda: (blur_h_kernel[grid, block](d_in, d_tmp, d_k1),
                         blur_v_composite_kernel[grid, block](
                             d_tmp, d_in, d_mask, d_out, d_k1)))
    naive = timed(lambda: blur_2d_kernel[grid, block](d_in, d_out, d_k2))
    return {'separable_ms': sep * 1e3, 'naive_2d_ms': naive * 1e3,
            'speedup': naive / max(sep, 1e-9)}
