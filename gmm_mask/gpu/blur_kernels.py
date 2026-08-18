"""Kernel 2 — separable Gaussian blur fused with the composite.

    frame (H,W,3) u8  --blur_h-->  (H,W,3) u16  --blur_v + select-->  (H,W,3) u8
                            \\                                    /
                             `-------- mask, original pixel ------'

This replaces `cv2.GaussianBlur` + `cv2.copyTo` in `utils/post_processing.py`,
which is where the frame budget was going once v1/v2 moved the mask onto the
device: the model kernel had stopped being the bottleneck and the host had not.

**The blur is integer arithmetic, because OpenCV's is.**

`cv2.GaussianBlur` on a `CV_8U` image is not a floating-point convolution. It
quantises `getGaussianKernel` to Q8 and runs fixed point, and the quantiser is
*cumulative* rather than per-tap:

    KQ[i] = round(256*cumsum(k)[i]) - round(256*cumsum(k)[i-1])

At (15, 5.0) that is [9,11,15,17,19,22,23,24,23,22,19,17,15,11,9]; rounding
each tap on its own gives [9,11,14,17,20,...] and is wrong at taps 2 and 4.
Getting this right is the difference between matching OpenCV exactly and
matching it on 84% of pixels: a float64 convolution with correct rounding still
disagrees with `cv2` on about 16% of pixels, by one grey level. That 16% is
OpenCV's own departure from an ideal Gaussian, and reproducing OpenCV means
reproducing it.

Measured against cv2 4.13.0 and 5.0.0, over random frames, all-zero, all-255,
border-lit and non-tile-multiple sizes: **zero pixels differ**. Because there is
no floating-point arithmetic anywhere in this path, that equality also does not
depend on the GPU architecture or on compiler flags — unlike the MOG2 kernel,
whose float32 parity depends on how FMAs contract. This is the strongest
correctness claim in the project.

`sum(KQ) == 256` is the invariant everything rests on. It is what keeps the
horizontal accumulator below 65536, and what makes the final uint8 saturation
unreachable, so neither pass needs a clamp. The host helper asserts it.

**Why two kernels and not one.** A single fused kernel would stage a
(TILE_Y+14) x (TILE_X+14) source tile, compute the horizontal pass into a
second shared tile, then reduce vertically — saving the write and re-read of
the 12.44 MB intermediate (~78 us at 1080p) at the cost of recomputing the
horizontal pass for the halo rows (+88% horizontal MACs, ~33 us). Roughly a
wash, for a materially more complex kernel. Two kernels, and the arithmetic is
written down here so the choice is on the record as measured rather than
assumed.
"""
import numpy as np
from numba import cuda, uint8, uint16, uint32, int32, float32

from settings import BLUR_KSIZE, BLUR_SIGMA

# The blur wants its own block shape. The model's TILE_Y=8 is the wrong
# geometry here: a radius-7 halo is 14 rows, nearly two full tiles, so a
# 32x8 block would load 22 rows to produce 8 (2.75x redundancy). At 32x16 the
# halo is smaller than the tile and the redundancy falls to 1.875x.
BLUR_TILE_X = 32
BLUR_TILE_Y = 16
BLUR_R = int(BLUR_KSIZE) // 2          # 7
BLUR_KLEN = int(BLUR_KSIZE)

_SH_H_W = BLUR_TILE_X + 2 * BLUR_R     # 46
_SH_V_H = BLUR_TILE_Y + 2 * BLUR_R     # 30


# ── host helpers: the specification, in numpy ─────────────────────────────────

def gaussian_kernel_q8(ksize: int = BLUR_KSIZE, sigma: float = BLUR_SIGMA):
    """OpenCV's fixed-point Gaussian kernel, derived rather than copied.

    Derived, so that changing BLUR_SIGMA changes the kernel instead of silently
    doing nothing — which is what baking the coefficients in as a literal would
    have done.
    """
    ksize = int(ksize)
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError(f"ksize must be odd and >= 3, got {ksize}")
    if float(sigma) <= 0.0:
        raise ValueError(
            f"sigma must be > 0, got {sigma}. OpenCV substitutes a hardcoded "
            "binomial kernel for sigma <= 0 and this derivation would not "
            "match it.")

    import cv2
    k = cv2.getGaussianKernel(ksize, float(sigma)).ravel()
    cum = np.round(256.0 * np.cumsum(k))
    kq = np.diff(np.concatenate(([0.0], cum))).astype(np.int32)

    # Load-bearing, not decorative: this is what bounds the accumulators.
    assert kq.sum() == 256, f"Q8 kernel sums to {kq.sum()}, not 256"
    assert (kq >= 0).all(), "negative tap — not a Gaussian"
    return kq


def blur_reference(frame_bgr: np.ndarray, kq: np.ndarray) -> np.ndarray:
    """The kernels' arithmetic written in numpy — an executable spec.

    Verified bit-exact against `cv2.GaussianBlur`, which lets a GPU-free test
    pin the claim and localises a kernel disagreement to the kernel instead of
    confusing it with an OpenCV version change.
    """
    import cv2
    r = len(kq) // 2
    h, w = frame_bgr.shape[:2]

    pad = cv2.copyMakeBorder(frame_bgr, 0, 0, r, r,
                             cv2.BORDER_REFLECT_101).astype(np.uint32)
    hor = sum(int(kq[j]) * pad[:, j:j + w] for j in range(len(kq)))
    assert hor.max() < 65536, "horizontal accumulator overflowed uint16"

    padv = cv2.copyMakeBorder(hor.astype(np.int32), r, r, 0, 0,
                              cv2.BORDER_REFLECT_101).astype(np.uint64)
    acc = sum(int(kq[j]) * padv[j:j + h] for j in range(len(kq)))
    return ((acc + 32768) >> 16).astype(np.uint8)


def blur_grid_for(height, width):
    return ((int(width) + BLUR_TILE_X - 1) // BLUR_TILE_X,
            (int(height) + BLUR_TILE_Y - 1) // BLUR_TILE_Y)


# ── device helpers ────────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def _reflect101(i, n):
    """BORDER_REFLECT_101: the border pixel is not repeated (abc -> cb|abc|ba).

    The `n == 1` guard is not defensive padding. Without it the loop below
    never terminates on a single-row or single-column image: -7 reflects to 7,
    7 reflects back to 2*(1-1)-7 = -7, forever. There is a test for exactly
    this, because nothing else in the pipeline would ever reach it.
    """
    if n == 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        if i >= n:
            i = 2 * (n - 1) - i
    return i


# ── v1: one kernel per pass, straight from global memory ─────────────────────

@cuda.jit
def blur_h_kernel(src_bgr, dst_q8, kq):
    """Horizontal pass. uint8 in, Q8-scaled uint16 out, no rounding yet.

    Rounding here instead of once at the end is what separates a 'Gaussian
    blur' from *this* Gaussian blur: OpenCV carries the horizontal result at
    full Q8 precision and rounds once, after the vertical pass.
    """
    x, y = cuda.grid(2)
    H, W = src_bgr.shape[0], src_bgr.shape[1]
    if y >= H or x >= W:
        return
    for c in range(3):
        acc = uint32(0)
        for j in range(BLUR_KLEN):
            nx = _reflect101(x + j - BLUR_R, W)
            acc += uint32(kq[j]) * uint32(src_bgr[y, nx, c])
        dst_q8[y, x, c] = uint16(acc)


@cuda.jit
def blur_v_composite_kernel(hor_q8, src_bgr, mask, dst_bgr, kq):
    """Vertical pass with the composite folded in.

    The select is free here and nowhere else: the blurred value is already in a
    register, the mask is one byte, and the only thing missing is the original
    pixel — one extra 3 B/px read that the horizontal kernel just left in L2.

    The mask test is hoisted out of the channel loop so a fully-background warp
    skips 45 multiply-accumulates outright. `!= 0`, not `== 255`, because that
    is `cv2.copyTo`'s rule; `fill_holes` happens to emit only 0 and 255 today
    and the kernel must not quietly depend on it.
    """
    x, y = cuda.grid(2)
    H, W = hor_q8.shape[0], hor_q8.shape[1]
    if y >= H or x >= W:
        return

    if mask[y, x] != uint8(0):
        for c in range(3):
            dst_bgr[y, x, c] = src_bgr[y, x, c]
        return

    for c in range(3):
        acc = uint32(0)
        for j in range(BLUR_KLEN):
            ny = _reflect101(y + j - BLUR_R, H)
            acc += uint32(kq[j]) * uint32(hor_q8[ny, x, c])
        dst_bgr[y, x, c] = uint8((acc + uint32(32768)) >> uint32(16))


# ── v2: shared-memory tiled ───────────────────────────────────────────────────

@cuda.jit
def blur_h_tiled_kernel(src_bgr, dst_q8, kq):
    """Same result as `blur_h_kernel`, reading each pixel once per block.

    The naive version loads 15 values per pixel per channel and every one of
    them is read by up to 14 neighbouring threads. A (TILE_Y, TILE_X+14, 3)
    tile turns those into shared loads plus about 1.44 global loads per pixel.
    """
    sh = cuda.shared.array((BLUR_TILE_Y, _SH_H_W, 3), uint8)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x * BLUR_TILE_X
    by = cuda.blockIdx.y * BLUR_TILE_Y
    x = bx + tx
    y = by + ty
    H, W = src_bgr.shape[0], src_bgr.shape[1]

    # Cooperative load, strided so the halo divides evenly across the block no
    # matter how it lines up with the tile width.
    gy = _reflect101(by + ty, H)
    for sx in range(tx, _SH_H_W, BLUR_TILE_X):
        gx = _reflect101(bx + sx - BLUR_R, W)
        for c in range(3):
            sh[ty, sx, c] = src_bgr[gy, gx, c]
    cuda.syncthreads()

    if y >= H or x >= W:
        return
    for c in range(3):
        acc = uint32(0)
        for j in range(BLUR_KLEN):
            acc += uint32(kq[j]) * uint32(sh[ty, tx + j, c])
        dst_q8[y, x, c] = uint16(acc)


@cuda.jit
def blur_v_composite_tiled_kernel(hor_q8, src_bgr, mask, dst_bgr, kq):
    """Tiled vertical pass, composite fused. See `blur_v_composite_kernel`.

    Note the early return cannot come before `syncthreads`: threads outside the
    image still have to take part in the cooperative load, or the rows they
    were responsible for are never staged and the tile is read uninitialised.
    """
    sh = cuda.shared.array((_SH_V_H, BLUR_TILE_X, 3), uint16)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x * BLUR_TILE_X
    by = cuda.blockIdx.y * BLUR_TILE_Y
    x = bx + tx
    y = by + ty
    H, W = hor_q8.shape[0], hor_q8.shape[1]

    gx = _reflect101(bx + tx, W)
    for sy in range(ty, _SH_V_H, BLUR_TILE_Y):
        gy = _reflect101(by + sy - BLUR_R, H)
        for c in range(3):
            sh[sy, tx, c] = hor_q8[gy, gx, c]
    cuda.syncthreads()

    if y >= H or x >= W:
        return

    if mask[y, x] != uint8(0):
        for c in range(3):
            dst_bgr[y, x, c] = src_bgr[y, x, c]
        return

    for c in range(3):
        acc = uint32(0)
        for j in range(BLUR_KLEN):
            acc += uint32(kq[j]) * uint32(sh[ty + j, tx, c])
        dst_bgr[y, x, c] = uint8((acc + uint32(32768)) >> uint32(16))


# ── colour conversion, so the bus carries 3 bytes a pixel and not 12 ─────────

@cuda.jit
def bgr2ycrcb_planar_kernel(src_bgr, dst, to_ycrcb):
    """(H,W,3) uint8 BGR -> (3,H,W) float32 model input, on the device.

    This is what makes the transfer story work. Today the host runs cvtColor,
    casts to float32 and transposes, then uploads 12 bytes a pixel; here the
    frame goes up as 3 bytes a pixel and the conversion is a per-pixel kernel.

    `cv2.cvtColor(BGR2YCrCb)` on uint8 is also integer, and is reproduced here
    exactly rather than approximated:

        Y  = D(1868*B + 9617*G + 4899*R)
        Cr = D((R-Y)*11682 + (128<<14))
        Cb = D((B-Y)*9241  + (128<<14))        D(x) = (x + (1<<13)) >> 14

    Exactly matters more than it might look. If the conversion drifted by a
    single grey level the model would see different input, and every quality
    number in the report would have been measured on the old pipeline. Because
    it is exact, the mask is provably unchanged — and `--parity-vs` in
    `eval_highway.py` is what proves it on all 1231 scored frames.

    `to_ycrcb=False` passes BGR through, for `--colorspace bgr`.
    """
    x, y = cuda.grid(2)
    H, W = src_bgr.shape[0], src_bgr.shape[1]
    if y >= H or x >= W:
        return

    b = int32(src_bgr[y, x, 0])
    g = int32(src_bgr[y, x, 1])
    r = int32(src_bgr[y, x, 2])

    if not to_ycrcb:
        dst[0, y, x] = float32(b)
        dst[1, y, x] = float32(g)
        dst[2, y, x] = float32(r)
        return

    yy = (1868 * b + 9617 * g + 4899 * r + 8192) >> 14
    cr = ((r - yy) * 11682 + (128 << 14) + 8192) >> 14
    cb = ((b - yy) * 9241 + (128 << 14) + 8192) >> 14
    dst[0, y, x] = float32(yy)
    dst[1, y, x] = float32(cr)
    dst[2, y, x] = float32(cb)


def warmup(height=BLUR_TILE_Y * 2, width=BLUR_TILE_X * 2):
    """Compile every kernel on a small frame so benchmarks exclude JIT time."""
    grid, block = blur_grid_for(height, width), (BLUR_TILE_X, BLUR_TILE_Y)
    kq = cuda.to_device(gaussian_kernel_q8())
    d_src = cuda.to_device(np.zeros((height, width, 3), np.uint8))
    d_hor = cuda.device_array((height, width, 3), np.uint16)
    d_out = cuda.device_array((height, width, 3), np.uint8)
    d_msk = cuda.to_device(np.zeros((height, width), np.uint8))
    d_pl = cuda.device_array((3, height, width), np.float32)

    blur_h_kernel[grid, block](d_src, d_hor, kq)
    blur_v_composite_kernel[grid, block](d_hor, d_src, d_msk, d_out, kq)
    blur_h_tiled_kernel[grid, block](d_src, d_hor, kq)
    blur_v_composite_tiled_kernel[grid, block](d_hor, d_src, d_msk, d_out, kq)
    bgr2ycrcb_planar_kernel[grid, block](d_src, d_pl, True)
    cuda.synchronize()
