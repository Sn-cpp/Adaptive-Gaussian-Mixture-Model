import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _dilate(mask_in, mask_out, H, W, radius):
    """Binary dilation: output[y,x]=255 iff any neighbor within `radius` is 255."""
    for y in prange(H):
        for x in range(W):
            found = False
            for dy in range(-radius, radius + 1):
                if found:
                    break
                ny = y + dy
                if ny < 0 or ny >= H:
                    continue
                for dx in range(-radius, radius + 1):
                    nx = x + dx
                    if 0 <= nx < W and mask_in[ny, nx] > np.uint8(0):
                        found = True
                        break
            mask_out[y, x] = np.uint8(255) if found else np.uint8(0)


@njit(parallel=True, cache=True)
def _erode(mask_in, mask_out, H, W, radius):
    """Binary erosion: output[y,x]=255 iff ALL neighbors within `radius` are 255."""
    for y in prange(H):
        for x in range(W):
            all_fg = True
            for dy in range(-radius, radius + 1):
                if not all_fg:
                    break
                ny = y + dy
                for dx in range(-radius, radius + 1):
                    nx = x + dx
                    if ny < 0 or ny >= H or nx < 0 or nx >= W:
                        all_fg = False   # treat out-of-bounds as background
                        break
                    if mask_in[ny, nx] == np.uint8(0):
                        all_fg = False
                        break
            mask_out[y, x] = np.uint8(255) if all_fg else np.uint8(0)


def morphological_close(mask, tmp, out, H, W, radius=3):
    """Dilation → erosion: fills holes up to `radius` pixels wide."""
    _dilate(mask, tmp, H, W, radius)
    _erode(tmp, out, H, W, radius)


def morphological_open(mask, tmp, out, H, W, radius=2):
    """Erosion → dilation: removes isolated blobs smaller than `radius` pixels."""
    _erode(mask, tmp, H, W, radius)
    _dilate(tmp, out, H, W, radius)


@njit(cache=True)
def largest_component(mask, H, W):
    """Keep only the largest 4-connected foreground component (sequential BFS).

    Returns a new (H, W) uint8 mask containing only the biggest blob.
    Use when the scene has one dominant foreground object.
    """
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
            head = np.int32(0)
            tail = np.int32(0)
            visited[sy, sx] = label_id
            queue[tail] = sy * W + sx
            tail += np.int32(1)
            while head < tail:
                n   = queue[head]; head += np.int32(1)
                ny0 = n // W;      nx0 = n % W
                for d in range(4):
                    ny = ny0 + DY[d]
                    nx = nx0 + DX[d]
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask[ny, nx] == np.uint8(255) and visited[ny, nx] == 0:
                            visited[ny, nx] = label_id
                            queue[tail] = ny * W + nx
                            tail += np.int32(1)
            sz = tail  # number of pixels in component
            if sz > best_sz:
                best_sz  = sz
                best_lbl = label_id

    out = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            if visited[y, x] == best_lbl:
                out[y, x] = np.uint8(255)
    return out


def warmup_morph(H=4, W=4, radius=1):
    """Pre-compile all morph kernels on tiny arrays."""
    m   = np.zeros((H, W), dtype=np.uint8)
    tmp = np.zeros((H, W), dtype=np.uint8)
    out = np.zeros((H, W), dtype=np.uint8)
    _dilate(m, tmp, H, W, radius)
    _erode(m, tmp, H, W, radius)
    morphological_close(m, tmp, out, H, W, radius)
    morphological_open(m, tmp, out, H, W, radius)
    largest_component(m, H, W)