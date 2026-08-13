"""Tiled Push-Relabel — GPU v1.

Each CUDA block owns one 16×16 pixel tile. Push-relabel inner passes run
entirely in shared memory for intra-tile pairs (no atomics).  Cross-tile
pushes use cuda.atomic.add on global arrays — at most 64 boundary pixels
per tile, so ~77 % of all push operations are atomic-free.

Global relabeling (height reset via BFS from sink) runs on CPU every
RELABEL_FREQ outer iterations, same cadence as the CPU v0.

Outer loop
──────────
  preflow → initial BFS → for each outer iteration:
      launch _tiled_pr_kernel (N_INNER passes / launch)
      every RELABEL_FREQ iters: CPU BFS → upload heights
  final BFS → extract labeling

Shared memory per block  (7 arrays × 256 floats or ints × 4 B = 7 KB)
  sh_excess, sh_height, sh_res_{right,down,left,up,snk}
"""

import numpy as np
from numba import cuda, njit
from time import perf_counter

_INF_H          = np.int32(2_000_000)
_TILE_H         = 16
_TILE_W         = 16
_N_INNER        = 8      # push-relabel passes per kernel launch
_DIR_RIGHT      = np.int32(0)
_DIR_DOWN       = np.int32(1)
_DIR_LEFT       = np.int32(2)
_DIR_UP         = np.int32(3)
_DIR_SNK        = np.int32(4)


# ── tiled push-relabel kernel ─────────────────────────────────────────────────

@cuda.jit
def _tiled_pr_kernel(res_right, res_down, res_left, res_up, res_snk,
                     excess, height, H, W, SINK):
    """One outer iteration: N_INNER push-relabel passes per tile in shared mem."""
    tx  = cuda.threadIdx.x   # 0..15
    ty  = cuda.threadIdx.y   # 0..15
    tid = ty * _TILE_W + tx  # 0..255

    gx = cuda.blockIdx.x * _TILE_W + tx
    gy = cuda.blockIdx.y * _TILE_H + ty
    valid = (gx < W) and (gy < H)
    n = gy * W + gx

    # ── shared memory ─────────────────────────────────────────────────────────
    sh_exc   = cuda.shared.array(shape=(256,), dtype=np.float32)
    sh_h     = cuda.shared.array(shape=(256,), dtype=np.int32)
    sh_rr    = cuda.shared.array(shape=(256,), dtype=np.float32)
    sh_rd    = cuda.shared.array(shape=(256,), dtype=np.float32)
    sh_rl    = cuda.shared.array(shape=(256,), dtype=np.float32)
    sh_ru    = cuda.shared.array(shape=(256,), dtype=np.float32)
    sh_rsnk  = cuda.shared.array(shape=(256,), dtype=np.float32)

    # ── load ──────────────────────────────────────────────────────────────────
    if valid:
        sh_exc [tid] = excess   [n]
        sh_h   [tid] = height   [n]
        sh_rr  [tid] = res_right[n]
        sh_rd  [tid] = res_down [n]
        sh_rl  [tid] = res_left [n]
        sh_ru  [tid] = res_up   [n]
        sh_rsnk[tid] = res_snk  [n]
    else:
        sh_exc [tid] = np.float32(0.0)
        sh_h   [tid] = _INF_H
        sh_rr  [tid] = np.float32(0.0)
        sh_rd  [tid] = np.float32(0.0)
        sh_rl  [tid] = np.float32(0.0)
        sh_ru  [tid] = np.float32(0.0)
        sh_rsnk[tid] = np.float32(0.0)
    cuda.syncthreads()

    h_sink = height[SINK]   # read once from global (rarely changes within loop)

    # ── inner passes ──────────────────────────────────────────────────────────
    for _ in range(_N_INNER):
        if valid and sh_exc[tid] > np.float32(1e-7):
            hn = sh_h[tid]
            if hn < _INF_H:
                min_h   = _INF_H
                min_dir = np.int32(-1)

                # right
                if gx + 1 < W and sh_rr[tid] > np.float32(0.0):
                    h_nbr = sh_h[tid + 1] if tx + 1 < _TILE_W else height[n + 1]
                    if h_nbr < min_h:
                        min_h = h_nbr; min_dir = _DIR_RIGHT

                # down
                if gy + 1 < H and sh_rd[tid] > np.float32(0.0):
                    h_nbr = sh_h[tid + _TILE_W] if ty + 1 < _TILE_H else height[n + W]
                    if h_nbr < min_h:
                        min_h = h_nbr; min_dir = _DIR_DOWN

                # left
                if gx > 0 and sh_rl[tid] > np.float32(0.0):
                    h_nbr = sh_h[tid - 1] if tx > 0 else height[n - 1]
                    if h_nbr < min_h:
                        min_h = h_nbr; min_dir = _DIR_LEFT

                # up
                if gy > 0 and sh_ru[tid] > np.float32(0.0):
                    h_nbr = sh_h[tid - _TILE_W] if ty > 0 else height[n - W]
                    if h_nbr < min_h:
                        min_h = h_nbr; min_dir = _DIR_UP

                # sink
                if sh_rsnk[tid] > np.float32(0.0) and h_sink < min_h:
                    min_h = h_sink; min_dir = _DIR_SNK

                if min_h >= _INF_H:
                    pass  # isolated — skip

                elif hn == min_h + np.int32(1):
                    # ── push ──────────────────────────────────────────────────
                    exc   = sh_exc[tid]
                    if min_dir == _DIR_SNK:
                        delta = exc if exc < sh_rsnk[tid] else sh_rsnk[tid]
                        sh_rsnk[tid]  -= delta
                        sh_exc [tid]  -= delta
                        cuda.atomic.add(excess, SINK, delta)

                    elif min_dir == _DIR_RIGHT:
                        delta = exc if exc < sh_rr[tid] else sh_rr[tid]
                        sh_rr [tid] -= delta
                        sh_exc[tid] -= delta
                        if tx + 1 < _TILE_W:
                            cuda.atomic.add(sh_exc, tid + 1,      delta)
                            cuda.atomic.add(sh_rl,  tid + 1,      delta)
                        else:
                            cuda.atomic.add(excess,   n + 1,      delta)
                            cuda.atomic.add(res_left, n + 1,      delta)

                    elif min_dir == _DIR_DOWN:
                        delta = exc if exc < sh_rd[tid] else sh_rd[tid]
                        sh_rd [tid] -= delta
                        sh_exc[tid] -= delta
                        if ty + 1 < _TILE_H:
                            cuda.atomic.add(sh_exc, tid + _TILE_W, delta)
                            cuda.atomic.add(sh_ru,  tid + _TILE_W, delta)
                        else:
                            cuda.atomic.add(excess, n + W,         delta)
                            cuda.atomic.add(res_up, n + W,         delta)

                    elif min_dir == _DIR_LEFT:
                        delta = exc if exc < sh_rl[tid] else sh_rl[tid]
                        sh_rl [tid] -= delta
                        sh_exc[tid] -= delta
                        if tx > 0:
                            cuda.atomic.add(sh_exc,    tid - 1,   delta)
                            cuda.atomic.add(sh_rr,     tid - 1,   delta)
                        else:
                            cuda.atomic.add(excess,    n - 1,     delta)
                            cuda.atomic.add(res_right, n - 1,     delta)

                    else:  # UP
                        delta = exc if exc < sh_ru[tid] else sh_ru[tid]
                        sh_ru [tid] -= delta
                        sh_exc[tid] -= delta
                        if ty > 0:
                            cuda.atomic.add(sh_exc, tid - _TILE_W, delta)
                            cuda.atomic.add(sh_rd,  tid - _TILE_W, delta)
                        else:
                            cuda.atomic.add(excess,  n - W,        delta)
                            cuda.atomic.add(res_down, n - W,       delta)

                else:
                    # ── relabel ───────────────────────────────────────────────
                    sh_h[tid] = min_h + np.int32(1)

        cuda.syncthreads()

    # ── write back ────────────────────────────────────────────────────────────
    if valid:
        excess   [n] = sh_exc [tid]
        height   [n] = sh_h   [tid]
        res_right[n] = sh_rr  [tid]
        res_down [n] = sh_rd  [tid]
        res_left [n] = sh_rl  [tid]
        res_up   [n] = sh_ru  [tid]
        res_snk  [n] = sh_rsnk[tid]


# ── active-excess reduction ───────────────────────────────────────────────────

@cuda.jit
def _max_excess_kernel(excess, result, N):
    sh  = cuda.shared.array(shape=(256,), dtype=np.float32)
    tid = cuda.threadIdx.x
    n   = cuda.grid(1)
    sh[tid] = excess[n] if n < N else np.float32(0.0)
    cuda.syncthreads()
    stride = np.int32(128)
    while stride > 0:
        if tid < stride and sh[tid + stride] > sh[tid]:
            sh[tid] = sh[tid + stride]
        cuda.syncthreads()
        stride >>= 1
    if tid == 0:
        cuda.atomic.max(result, 0, sh[0])


# ── CPU global relabeling (BFS from sink on reverse-residual graph) ───────────

@njit(cache=True)
def _global_relabel(res_right, res_down, res_left, res_up, res_snk,
                    H, W, height_label):
    N    = H * W
    SINK = N + np.int32(1)
    for i in range(N + 2):
        height_label[i] = _INF_H
    queue = np.empty(N + 2, dtype=np.int32)
    head  = np.int32(0); tail = np.int32(0)
    height_label[SINK] = np.int32(0)
    queue[tail] = SINK; tail += np.int32(1)
    while head < tail:
        u  = queue[head]; head += np.int32(1)
        hu = height_label[u]
        if u == SINK:
            for nb in range(N):
                if height_label[nb] == _INF_H and res_snk[nb] > np.float32(0.0):
                    height_label[nb] = hu + np.int32(1)
                    queue[tail] = nb; tail += np.int32(1)
        else:
            uy = u // W; ux = u % W
            if ux + np.int32(1) < W:
                v = u + np.int32(1)
                if height_label[v] == _INF_H and res_left[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v; tail += np.int32(1)
            if ux > np.int32(0):
                v = u - np.int32(1)
                if height_label[v] == _INF_H and res_right[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v; tail += np.int32(1)
            if uy + np.int32(1) < H:
                v = u + W
                if height_label[v] == _INF_H and res_up[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v; tail += np.int32(1)
            if uy > np.int32(0):
                v = u - W
                if height_label[v] == _INF_H and res_down[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v; tail += np.int32(1)
    height_label[N] = np.int32(N + 2)


# ── public API ────────────────────────────────────────────────────────────────

def push_relabel_tiled(cap_src, cap_snk, cap_right, cap_down,
                       H, W, max_iter=200, relabel_freq=25):
    """Tiled GPU push-relabel on a 4-connected pixel grid.

    Inputs: 1-D float32 device arrays of length N = H*W.
    Returns: (H,W) uint8 device array (255 = foreground, 0 = background).
    """

    N    = np.int32(H * W)
    SINK = N + np.int32(1)

    # Download caps once for initial CPU BFS — all four arrays in one pass
    h_res_r   = cap_right.copy_to_host()
    h_res_d   = cap_down .copy_to_host()
    h_res_snk = cap_snk  .copy_to_host()
    h_height  = np.zeros(int(N) + 2, dtype=np.int32)

    _global_relabel(h_res_r, h_res_d, h_res_r, h_res_d, h_res_snk,
                    H, W, h_height)

    # cuda.to_device always allocates fresh device memory, so uploading
    # h_res_r twice produces two independent device arrays for right/left.
    res_right = cuda.to_device(h_res_r)
    res_down  = cuda.to_device(h_res_d)
    res_left  = cuda.to_device(h_res_r)
    res_up    = cuda.to_device(h_res_d)
    res_snk   = cuda.to_device(h_res_snk)
    d_height  = cuda.to_device(h_height)

    h_excess = np.zeros(int(N) + 2, np.float32)
    h_excess[:int(N)] = cap_src.copy_to_host()
    excess = cuda.to_device(h_excess)

    # Kernel launch geometry
    tile_grid  = ((int(W) + _TILE_W - 1) // _TILE_W,
                  (int(H) + _TILE_H - 1) // _TILE_H)
    tile_block = (_TILE_W, _TILE_H)
    exc_block  = 256
    exc_grid   = (int(N) + exc_block - 1) // exc_block
    d_max_exc  = cuda.to_device(np.zeros(1, np.float32))
    zero_exc   = np.zeros(1, np.float32)


    for iteration in range(max_iter):
        _tiled_pr_kernel[tile_grid, tile_block](
            res_right, res_down, res_left, res_up, res_snk,
            excess, d_height, H, W, SINK)
        cuda.synchronize()

        d_max_exc.copy_to_device(zero_exc)
        _max_excess_kernel[exc_grid, exc_block](excess, d_max_exc, N)
        cuda.synchronize()


        if float(d_max_exc.copy_to_host()[0]) <= 1e-7:
            break
        
        if (iteration + 1) % relabel_freq == 0:
            h_res_r   = res_right.copy_to_host()
            h_res_d   = res_down .copy_to_host()
            h_res_l   = res_left .copy_to_host()
            h_res_u   = res_up   .copy_to_host()
            h_res_snk = res_snk  .copy_to_host()
            _global_relabel(h_res_r, h_res_d, h_res_l, h_res_u, h_res_snk,
                            H, W, h_height)
            d_height.copy_to_device(h_height)


    # Final BFS to extract cut
    h_res_r   = res_right.copy_to_host()
    h_res_d   = res_down .copy_to_host()
    h_res_l   = res_left .copy_to_host()
    h_res_u   = res_up   .copy_to_host()
    h_res_snk = res_snk  .copy_to_host()

    _global_relabel(h_res_r, h_res_d, h_res_l, h_res_u, h_res_snk,
                    H, W, h_height)

    # t1 = perf_counter()
    # print("Post ITR: ", (t1-t0)*1000)

    d_height.copy_to_device(h_height)
    d_labeling = cuda.device_array((int(H), int(W)), dtype=np.uint8)

    make_labeling[exc_grid, exc_block](
        d_height,
        d_labeling,
        H, W,
        _INF_H
    )
    cuda.synchronize()

    return d_labeling

@cuda.jit
def make_labeling(h_height, d_labeling, H, W, INF_H):
    n = cuda.grid(1)

    if n < H * W:
        y = n // W
        x = n - y * W

        if h_height[n] >= INF_H:
            d_labeling[y, x] = np.uint8(255)
        else:
            d_labeling[y, x] = np.uint8(0)

def warmup_push_relabel_tiled():
    H, W = np.int32(16), np.int32(16)
    N = H * W
    push_relabel_tiled(
        cuda.to_device(np.zeros(int(N), np.float32)),
        cuda.to_device(np.zeros(int(N), np.float32)),
        cuda.to_device(np.zeros(int(N), np.float32)),
        cuda.to_device(np.zeros(int(N), np.float32)),
        H, W, max_iter=2, relabel_freq=1,
    )
