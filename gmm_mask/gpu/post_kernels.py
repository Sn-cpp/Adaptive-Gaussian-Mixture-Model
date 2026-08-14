"""Post-processing as CUDA kernels — the chain `eval_highway.py` picked.

    bg_prob  --threshold-->  binary mask  --median 5-->  refined mask
                                                              |
                                        host flood fill  <----+

Two facts make this cheap enough to be worth moving off the host at all.

**A median of a binary image is a majority vote.** `cv2.medianBlur` on a mask
whose values are only 0 and 255 returns 255 exactly when at least 13 of the 25
window values are 255. So there is no sorting network here: count the ones and
compare. That is 25 loads and an integer add per pixel, and it is *bit-exact*
with OpenCV rather than an approximation of it.

**The threshold is free.** It reads a value the MOG2 kernel already wrote and
writes one byte. In `v2` it disappears entirely into that kernel's epilogue.

`fill_holes` stays on the host: it is a scan-line flood fill, and its
data-parallel equivalent (morphological reconstruction) needs one dilate per
pixel of propagation distance — 344 ms against 2.2 ms at 1080p. Profiling
says leave it, so it is left.
"""
import numpy as np
from numba import cuda, float32, int32, uint8

TILE_X = 32
TILE_Y = 8
HALO = 2                    # a 5x5 window reaches 2 pixels either side
MEDIAN_WIN = 25
MEDIAN_MAJORITY = 13        # ceil(25 / 2)


# ── v1: one kernel per stage ─────────────────────────────────────────────────

@cuda.jit
def threshold_kernel(bg_prob, out, thresh):
    """Foreground where the matched background modes carry too little weight."""
    x, y = cuda.grid(2)
    H, W = bg_prob.shape
    if y >= H or x >= W:
        return
    out[y, x] = uint8(255) if bg_prob[y, x] < thresh else uint8(0)


@cuda.jit
def median5_kernel(src, dst):
    """5x5 median of a binary mask, straight from global memory.

    Border handling is OpenCV's `BORDER_REPLICATE`, which is what `medianBlur`
    uses; getting this wrong shows up as a one-pixel frame around the mask and
    nowhere else, so it is easy to ship by accident.
    """
    x, y = cuda.grid(2)
    H, W = src.shape
    if y >= H or x >= W:
        return

    count = int32(0)
    for dy in range(-HALO, HALO + 1):
        ny = min(max(y + dy, 0), H - 1)
        for dx in range(-HALO, HALO + 1):
            nx = min(max(x + dx, 0), W - 1)
            if src[ny, nx] == uint8(255):
                count += 1
    dst[y, x] = uint8(255) if count >= MEDIAN_MAJORITY else uint8(0)


# ── v2: shared-memory tiled median ───────────────────────────────────────────

@cuda.jit
def median5_tiled_kernel(src, dst):
    """Same result as `median5_kernel`, reading each pixel once per block.

    The naive version loads 25 values per pixel; every one of them is also read
    by up to 24 neighbouring threads. Staging a (TILE_Y+4) x (TILE_X+4) tile in
    shared memory turns those 25 global loads into 25 shared loads plus roughly
    1.4 global loads per pixel.
    """
    sh = cuda.shared.array((TILE_Y + 2 * HALO, TILE_X + 2 * HALO), uint8)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    x = cuda.blockIdx.x * TILE_X + tx
    y = cuda.blockIdx.y * TILE_Y + ty
    H, W = src.shape

    # Cooperative load of the tile plus its halo. Strided so every thread does
    # its share regardless of how the halo divides into the block.
    for sy in range(ty, TILE_Y + 2 * HALO, TILE_Y):
        gy = min(max(cuda.blockIdx.y * TILE_Y + sy - HALO, 0), H - 1)
        for sx in range(tx, TILE_X + 2 * HALO, TILE_X):
            gx = min(max(cuda.blockIdx.x * TILE_X + sx - HALO, 0), W - 1)
            sh[sy, sx] = src[gy, gx]
    cuda.syncthreads()

    if y >= H or x >= W:
        return

    count = int32(0)
    for dy in range(2 * HALO + 1):
        for dx in range(2 * HALO + 1):
            if sh[ty + dy, tx + dx] == uint8(255):
                count += 1
    dst[y, x] = uint8(255) if count >= MEDIAN_MAJORITY else uint8(0)


# ── optional CLOSE, kept for footage where the subject is large ──────────────

@cuda.jit
def dilate_kernel(src, dst, radius):
    x, y = cuda.grid(2)
    H, W = src.shape
    if y >= H or x >= W:
        return
    v = uint8(0)
    for dy in range(-radius, radius + 1):
        ny = min(max(y + dy, 0), H - 1)
        for dx in range(-radius, radius + 1):
            nx = min(max(x + dx, 0), W - 1)
            if src[ny, nx] == uint8(255):
                v = uint8(255)
    dst[y, x] = v


@cuda.jit
def erode_kernel(src, dst, radius):
    x, y = cuda.grid(2)
    H, W = src.shape
    if y >= H or x >= W:
        return
    v = uint8(255)
    for dy in range(-radius, radius + 1):
        ny = min(max(y + dy, 0), H - 1)
        for dx in range(-radius, radius + 1):
            nx = min(max(x + dx, 0), W - 1)
            if src[ny, nx] != uint8(255):
                v = uint8(0)
    dst[y, x] = v


def grid_for(height, width):
    return ((int(width) + TILE_X - 1) // TILE_X,
            (int(height) + TILE_Y - 1) // TILE_Y)


def warmup(height=8, width=8):
    """Compile every kernel on tiny arrays so benchmarks exclude JIT time."""
    grid, block = grid_for(height, width), (TILE_X, TILE_Y)
    d_p = cuda.to_device(np.zeros((height, width), np.float32))
    d_a = cuda.to_device(np.zeros((height, width), np.uint8))
    d_b = cuda.device_array((height, width), np.uint8)
    threshold_kernel[grid, block](d_p, d_a, np.float32(0.5))
    median5_kernel[grid, block](d_a, d_b)
    median5_tiled_kernel[grid, block](d_a, d_b)
    dilate_kernel[grid, block](d_a, d_b, int32(1))
    erode_kernel[grid, block](d_a, d_b, int32(1))
    cuda.synchronize()
