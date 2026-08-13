"""Tile-Wave Push-Relabel — GPU v2.

Extends the tiled GPU push-relabel (v1) with a GPU-side global relabeling
pass that replaces the CPU BFS + PCIe round-trip.

Global relabeling (GPU BFS from sink)
──────────────────────────────────────
Instead of downloading all six residual arrays to CPU, running BFS, and
re-uploading height labels, v2 runs a vertex-parallel BFS directly on GPU:

  Repeat until converged:
    Each thread owns one pixel n.
    new_h = min over residual arcs (n→m) of  h[m] + 1.
    If new_h < h[n]: update h[n], set changed flag.

One kernel launch propagates the BFS frontier exactly one hop outward.
A global `changed` flag lets the outer loop terminate as soon as no pixel
updated — so the BFS stops after the actual diameter of the sink-reachable
subgraph, not the worst-case grid diameter.

Cost estimate vs CPU BFS and BF-wave (640×480, typical GrabCut)
  CPU BFS      : 7.4 MB download + ~0.5 ms BFS + upload ≈ 1.5 ms + sync
  BF-wave v2a  : 40 fixed launches × ~0.5 ms               ≈ 2.0 ms
  GPU BFS v2   : ~60 launches × ~0.1 ms + ~15 sync checks  ≈ 0.7 ms
  Early exit shrinks the launch count when the foreground blob is small.

Push-relabel kernel is identical to v1 (_tiled_pr_kernel, reused by import).
"""

import numpy as np
from time import perf_counter
from numba import cuda

from grabcut.gpu.cuda_v1.push_relabel_cuda_v1 import (
    _tiled_pr_kernel, _max_excess_kernel, _INF_H,
    _TILE_H, _TILE_W, _N_INNER,
)

# How many BFS steps to run between convergence checks.
# Each check costs one cuda.synchronize() + 4-byte D2H (~0.05 ms).
# 8 steps/check → at most ceil((H+W)/8) ≈ 140 checks worst-case.
_BFS_CHECK_EVERY = 8


# ── GPU vertex-parallel BFS (one hop per launch) ──────────────────────────────

@cuda.jit
def _bfs_step_kernel(res_right, res_down, res_left, res_up, res_snk,
                     height, changed, H, W, N):
    """One vertex-parallel BFS hop: relax h[n] = min(h[nbr]+1) over residual arcs.

    Sets changed[0] = 1 when any height decreased.
    Heights only decrease (monotone), so the loop converges in exactly
    diameter(sink-reachable subgraph) launches.
    """
    n = cuda.grid(1)
    if n >= N:
        return

    gx = n % W
    gy = n // W
    best = _INF_H

    # arc n → n+1 (right)
    if gx + np.int32(1) < W and res_right[n] > np.float32(0.0):
        h_nbr = height[n + np.int32(1)]
        cand  = h_nbr + np.int32(1)
        if cand < best: best = cand

    # arc n → n-1 (left)
    if gx > 0 and res_left[n] > np.float32(0.0):
        h_nbr = height[n - np.int32(1)]
        cand  = h_nbr + np.int32(1)
        if cand < best: best = cand

    # arc n → n+W (down)
    if gy + np.int32(1) < H and res_down[n] > np.float32(0.0):
        h_nbr = height[n + W]
        cand  = h_nbr + np.int32(1)
        if cand < best: best = cand

    # arc n → n-W (up)
    if gy > 0 and res_up[n] > np.float32(0.0):
        h_nbr = height[n - W]
        cand  = h_nbr + np.int32(1)
        if cand < best: best = cand

    # arc n → SINK: height[SINK] = 0 → candidate = 1
    if res_snk[n] > np.float32(0.0):
        if np.int32(1) < best: best = np.int32(1)

    if best < height[n]:
        height[n] = best
        changed[0] = np.int32(1)


def _gpu_global_relabel(res_right, res_down, res_left, res_up, res_snk,
                        d_height, H, W, SINK):
    """Reset height labels via vertex-parallel GPU BFS from sink.

    Terminates as soon as the BFS frontier is exhausted (changed flag clears),
    which is typically much earlier than the worst-case grid diameter.
    """
    n_total = int(H) * int(W) + 2
    h_init  = np.full(n_total, int(_INF_H), dtype=np.int32)
    h_init[int(SINK)]       = 0
    h_init[int(H) * int(W)] = int(H) * int(W) + 2  # SOURCE stays above SINK
    d_height.copy_to_device(h_init)

    N   = np.int32(H * W)
    blk = 256
    grid = (int(N) + blk - 1) // blk

    d_changed = cuda.to_device(np.zeros(1, np.int32))
    h_zero    = np.zeros(1, np.int32)

    max_iters = int(H) + int(W)  # tight upper bound on grid BFS diameter
    n_outer   = (max_iters + _BFS_CHECK_EVERY - 1) // _BFS_CHECK_EVERY

    for _ in range(n_outer):
        d_changed.copy_to_device(h_zero)
        for _ in range(_BFS_CHECK_EVERY):
            _bfs_step_kernel[grid, blk](
                res_right, res_down, res_left, res_up, res_snk,
                d_height, d_changed, H, W, N)
        cuda.synchronize()
        if d_changed.copy_to_host()[0] == 0:
            break  # BFS frontier exhausted — all reachable heights are exact



# ── public API ─────────────────────────────────────────────────────────────────

def push_relabel_wave(d_cap_src, d_cap_snk, d_cap_right, d_cap_down,
                      H, W, max_iter=200, relabel_freq=25):
    """Tile-wave GPU push-relabel on a 4-connected pixel grid.

    Identical contract to push_relabel_tiled (v1) but global relabeling runs
    entirely on GPU — no residual array download between relabel events.

    Inputs: 1-D float32 device arrays of length N = H*W.
    Returns: 1-D int32 labeling (0 = foreground, 1 = background).
    """
    N    = np.int32(H * W)
    SINK = N + np.int32(1)

    # cuda.to_device always allocates fresh device memory — uploading the same
    # host buffer twice gives two independent device arrays (res_right vs res_left).
    h_right = d_cap_right.copy_to_host()
    h_down  = d_cap_down .copy_to_host()
    res_right = cuda.to_device(h_right)
    res_down  = cuda.to_device(h_down)
    res_left  = cuda.to_device(h_right)
    res_up    = cuda.to_device(h_down)
    res_snk   = cuda.to_device(d_cap_snk.copy_to_host())

    h_excess = np.zeros(int(N) + 2, np.float32)
    h_excess[:int(N)] = d_cap_src.copy_to_host()
    excess = cuda.to_device(h_excess)

    # Initial height allocation via GPU wave
    d_height = cuda.to_device(np.zeros(int(N) + 2, np.int32))

    _gpu_global_relabel(res_right, res_down, res_left, res_up, res_snk,
                        d_height, H, W, SINK)


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
            _gpu_global_relabel(res_right, res_down, res_left, res_up, res_snk,
                                d_height, H, W, SINK)

    # Final GPU relabeling to extract cut labels
    _gpu_global_relabel(res_right, res_down, res_left, res_up, res_snk,
                        d_height, H, W, SINK)
    cuda.synchronize()

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

def warmup_push_relabel_wave():
    H, W = np.int32(16), np.int32(16)
    N = H * W
    push_relabel_wave(
        cuda.to_device(np.zeros(int(N), np.float32)),
        cuda.to_device(np.zeros(int(N), np.float32)),
        cuda.to_device(np.zeros(int(N), np.float32)),
        cuda.to_device(np.zeros(int(N), np.float32)),
        H, W, max_iter=2, relabel_freq=1,
    )
