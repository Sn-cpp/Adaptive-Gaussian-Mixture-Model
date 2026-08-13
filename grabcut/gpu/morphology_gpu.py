"""GPU morphology + Gaussian blur — shared by cuda_v1 and cuda_v2.

API mirrors morphology_cuda_v0 but operates on device arrays throughout.

  morphological_close(d_mask, d_tmp, d_out, H, W, radius=3)
  morphological_open (d_mask, d_tmp, d_out, H, W, radius=2)
  largest_component  (d_mask, H, W) -> (H,W) uint8 host array
  gaussian_blur_f32  (d_src, d_dst, H, W, ksize, sigma)

Design notes
────────────
Dilation / erosion  — simple per-pixel kernel; each thread checks a
  (2r+1)² neighbourhood. For r=3 that's 49 reads — fits in registers,
  no shared memory needed.  Two passes implement one dilate+erode operation.

Gaussian blur  — separable 1-D convolution.  Precomputed host weights are
  uploaded once and reused every frame.  Horizontal pass writes to a
  float32 intermediate; vertical pass writes the final result.

Largest connected component  — kept as a *hybrid*: after GPU morphological
  operations the (H,W) uint8 mask is small (≈300 kB for 640×480) so a
  fast download → @njit BFS → upload costs ≈ 0.6 ms — cheaper than
  implementing GPU BFS with its O(diameter) kernel-launch overhead.
"""

import math
import numpy as np
from numba import cuda, njit

# ── constants ──────────────────────────────────────────────────────────────────
_BLOCK2D = (16, 16)     # 2-D thread block for per-pixel kernels


# ── binary dilation ───────────────────────────────────────────────────────────

@cuda.jit
def _dilate_kernel(src, dst, H, W, radius):
    y, x = cuda.grid(2)
    if y >= H or x >= W:
        return
    if src[y, x] > np.uint8(0):
        dst[y, x] = np.uint8(255)
        return
    found = False
    for dy in range(-radius, radius + 1):
        if found:
            break
        ny = y + dy
        if ny < 0 or ny >= H:
            continue
        for dx in range(-radius, radius + 1):
            nx = x + dx
            if 0 <= nx < W and src[ny, nx] > np.uint8(0):
                found = True
                break
    dst[y, x] = np.uint8(255) if found else np.uint8(0)


# ── binary erosion ─────────────────────────────────────────────────────────────

@cuda.jit
def _erode_kernel(src, dst, H, W, radius):
    y, x = cuda.grid(2)
    if y >= H or x >= W:
        return
    all_fg = True
    for dy in range(-radius, radius + 1):
        if not all_fg:
            break
        ny = y + dy
        for dx in range(-radius, radius + 1):
            nx = x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                all_fg = False
                break
            if src[ny, nx] == np.uint8(0):
                all_fg = False
                break
    dst[y, x] = np.uint8(255) if all_fg else np.uint8(0)


# ── morphological ops ─────────────────────────────────────────────────────────

def _make_grid2d(H, W):
    bx = (W + _BLOCK2D[1] - 1) // _BLOCK2D[1]
    by = (H + _BLOCK2D[0] - 1) // _BLOCK2D[0]
    return (int(by), int(bx))


def morphological_close(d_mask, d_tmp, d_out, H, W, radius=3):
    """Dilation → erosion on device arrays."""
    g = _make_grid2d(H, W)

    _dilate_kernel[g, _BLOCK2D](d_mask, d_tmp, H, W, radius)
    _erode_kernel [g, _BLOCK2D](d_tmp,  d_out, H, W, radius)


def morphological_open(d_mask, d_tmp, d_out, H, W, radius=2):
    """Erosion → dilation on device arrays."""
    g = _make_grid2d(H, W)
    _erode_kernel [g, _BLOCK2D](d_mask, d_tmp, H, W, radius)
    _dilate_kernel[g, _BLOCK2D](d_tmp,  d_out, H, W, radius)


# ── largest connected component (hybrid) ──────────────────────────────────────

@njit(cache=True)
def _bfs_largest(mask, H, W):
    """CPU BFS to find and return the largest 4-connected component."""
    visited  = np.zeros((H, W), dtype=np.int32)
    queue    = np.empty(H * W, dtype=np.int32)
    label_id = np.int32(0)
    best_sz  = np.int32(0)
    best_lbl = np.int32(0)

    DY = (-1, 1,  0, 0)
    DX = ( 0, 0, -1, 1)

    for sy in range(H):
        for sx in range(W):
            if mask[sy, sx] == np.uint8(0) or visited[sy, sx] != 0:
                continue
            label_id += np.int32(1)
            head = np.int32(0); tail = np.int32(0)
            visited[sy, sx] = label_id
            queue[tail] = sy * W + sx; tail += np.int32(1)
            while head < tail:
                n   = queue[head]; head += np.int32(1)
                ny0 = n // W;     nx0 = n % W
                for d in range(4):
                    ny = ny0 + DY[d]; nx = nx0 + DX[d]
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask[ny, nx] == np.uint8(255) and visited[ny, nx] == 0:
                            visited[ny, nx] = label_id
                            queue[tail] = ny * W + nx; tail += np.int32(1)
            sz = tail
            if sz > best_sz:
                best_sz = sz; best_lbl = label_id

    out = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            if visited[y, x] == best_lbl:
                out[y, x] = np.uint8(255)
    return out


def largest_component(d_mask, H, W):
    """Hybrid: download d_mask → CPU BFS → upload result back.

    Returns the same d_mask device array (updated in-place) for chaining.
    Round-trip cost ≈ 0.06 ms DMA + 0.5 ms BFS for 640×480.
    """
    h_mask = d_mask.copy_to_host()
    h_out  = _bfs_largest(h_mask, H, W)
    d_mask.copy_to_device(h_out)
    return d_mask


# ── Gaussian blur (separable, float32, 3-channel) ─────────────────────────────

@cuda.jit
def _gauss_h_kernel(src, tmp, weights, ksize, H, W):
    """Horizontal 1-D Gaussian pass: src(H,W,3) → tmp(H,W,3)."""
    y, x = cuda.grid(2)
    if y >= H or x >= W:
        return
    half = ksize // 2
    for c in range(3):
        acc = np.float32(0.0)
        for k in range(ksize):
            nx = x + k - half
            if 0 <= nx < W:
                acc += weights[k] * src[y, nx, c]
        tmp[y, x, c] = acc


@cuda.jit
def _gauss_v_kernel(tmp, dst, weights, ksize, H, W):
    """Vertical 1-D Gaussian pass: tmp(H,W,3) → dst(H,W,3)."""
    y, x = cuda.grid(2)
    if y >= H or x >= W:
        return
    half = ksize // 2
    for c in range(3):
        acc = np.float32(0.0)
        for k in range(ksize):
            ny = y + k - half
            if 0 <= ny < H:
                acc += weights[k] * tmp[ny, x, c]
        dst[y, x, c] = acc


def _make_gauss_weights(ksize, sigma):
    """1-D Gaussian kernel, normalised to sum=1."""
    half = ksize // 2
    w = np.array([math.exp(-0.5 * ((i - half) / sigma) ** 2)
                  for i in range(ksize)], dtype=np.float32)
    return (w / w.sum()).astype(np.float32)


# Cache compiled weights per (ksize, sigma) pair so we don't recompute.
_gauss_weight_cache = {}

def gaussian_blur_f32(d_src, d_tmp, d_dst, H, W, ksize=15, sigma=5.0):
    """In-place separable Gaussian blur on device float32 (H,W,3) arrays.

    d_tmp must be a pre-allocated (H,W,3) float32 device array (scratch).
    Writes result into d_dst.
    """
    key = (ksize, sigma)
    if key not in _gauss_weight_cache:
        _gauss_weight_cache[key] = cuda.to_device(_make_gauss_weights(ksize, sigma))
    d_weights = _gauss_weight_cache[key]

    g = _make_grid2d(H, W)
    _gauss_h_kernel[g, _BLOCK2D](d_src, d_tmp, d_weights, ksize, H, W)
    _gauss_v_kernel[g, _BLOCK2D](d_tmp, d_dst, d_weights, ksize, H, W)


# ── warmup ────────────────────────────────────────────────────────────────────

def warmup_morph_gpu(H=4, W=4, radius=1):
    """Pre-compile all GPU morph + blur kernels on tiny dummy data."""
    d_m   = cuda.to_device(np.zeros((H, W),    dtype=np.uint8))
    d_t   = cuda.to_device(np.zeros((H, W),    dtype=np.uint8))
    d_o   = cuda.to_device(np.zeros((H, W),    dtype=np.uint8))
    d_f   = cuda.to_device(np.zeros((H, W, 3), dtype=np.float32))
    d_ft  = cuda.to_device(np.zeros((H, W, 3), dtype=np.float32))
    d_fd  = cuda.to_device(np.zeros((H, W, 3), dtype=np.float32))
    morphological_close(d_m, d_t, d_o, H, W, radius)
    morphological_open (d_m, d_t, d_o, H, W, radius)
    largest_component  (d_m, H, W)
    gaussian_blur_f32  (d_f, d_ft, d_fd, H, W, ksize=3, sigma=1.0)
    cuda.synchronize()
