"""WBPR-style GPU push-relabel for a 4-connected image grid.

Algorithm overview (adapted from Hong et al., "Work-Balanced Parallel Push-Relabel
for the GPU", IPDPS 2023):

  Outer loop (Python):
    1. [GPU]  Preflow: push all cap_src to neighbours, set source height = N+2.
    2. while active excess > 0:
         a. [GPU]  scan_active_vertices  — build AVQ via atomic_add
         b. [GPU]  tiled_push_relabel    — TLPNS: one tile per active vertex,
                                           TILE_SIZE=4 threads per tile (one per
                                           direction), shared-mem reduction for
                                           MHNS, delegated thread does push/relabel
         c. every RELABEL_FREQ iters:
            [CPU]  global_relabel via BFS from sink on reverse-residual graph
    3. [CPU]  Final BFS labeling → extract source-side component.

For image grids degree is exactly 4 (right, down, left, up) so TILE_SIZE=4
is the natural mapping. No RCSR needed — reverse edges are structurally known:
  left[n]  ↔ right[n-1],  up[n] ↔ down[n-W].
"""

import warnings

import numpy as np
from ...reverse_arcs import reverse_arcs
from numba import cuda, njit

# ── constants ─────────────────────────────────────────────────────────────────
_INF_H   = np.int32(2_000_000)
_TILE    = 4           # threads per tile: right, down, left, up
_BLOCK   = 128         # threads per CUDA block  (must be divisible by _TILE)
_TILES_PER_BLOCK = _BLOCK // _TILE   # = 32 tiles/block
_SHARED_MEMORY_SIZE = _TILES_PER_BLOCK * _TILE

# Direction indices within a tile
_DIR_RIGHT = 0
_DIR_DOWN  = 1
_DIR_LEFT  = 2
_DIR_UP    = 3


# ── AVQ kernel ────────────────────────────────────────────────────────────────

@cuda.jit
def _scan_active_vertices(excess, height_label, avq, avq_size, N):
    """Build the Active Vertex Queue: one thread per pixel node.

    A node at _INF_H cannot reach the SINK through the residual graph, so
    _tiled_push_relabel finds no admissible neighbour and returns immediately.
    Enqueuing it anyway means the queue never empties and the loop runs out
    max_iter doing nothing — returning a cut extracted from an unfinished
    preflow. Its excess is flow that belongs back at the SOURCE, which phase
    two of push-relabel handles and the minimum cut does not depend on.
    """
    n = cuda.grid(1)
    if n >= N:
        return
    if excess[n] > np.float32(1e-7) and height_label[n] < _INF_H:
        pos = cuda.atomic.add(avq_size, 0, np.int32(1))
        avq[pos] = np.int32(n)


# ── tiled push-relabel kernel ─────────────────────────────────────────────────

@cuda.jit
def _tiled_push_relabel(
        avq, avq_size,
        excess, height_label,
        res_right, res_down, res_left, res_up, res_snk,
        H, W, N):
    """TLPNS: one tile of 4 threads processes one active vertex.

    Shared memory layout (per tile):
      sh_h[4]    — neighbour heights (INF if no residual / out-of-bounds)
      sh_nbr[4]  — flat neighbour index (-1 if invalid)
      sh_res[4]  — residual capacity toward that neighbour
    """

    tid  = cuda.threadIdx.x
    bid  = cuda.blockIdx.x
    lane = tid % _TILE            # 0..3
    tile = tid // _TILE           # which tile within this block
    global_tile = bid * _TILES_PER_BLOCK + tile

    SINK = N + np.int32(1)

    # Shared memory: 4 arrays of TILES_PER_BLOCK x TILE
    sh_h   = cuda.shared.array(shape=(_SHARED_MEMORY_SIZE,), dtype=np.int32)
    sh_nbr = cuda.shared.array(shape=(_SHARED_MEMORY_SIZE,), dtype=np.int32)
    sh_res = cuda.shared.array(shape=(_SHARED_MEMORY_SIZE,), dtype=np.float32)


    base = tile * _TILE + lane    # index into shared arrays for this lane

    if global_tile >= avq_size[0]:
        return

    n  = avq[global_tile]
    hn = height_label[n]
    y  = n // W
    x  = n % W

    # Step 1: each lane gathers one neighbour's (height, index, residual)
    nbr_idx = np.int32(-1)
    nbr_res = np.float32(0.0)
    nbr_h   = _INF_H

    if lane == _DIR_RIGHT:
        if x + np.int32(1) < W:
            v = n + np.int32(1)
            r = res_right[n]
            if r > np.float32(0.0):
                nbr_idx = v
                nbr_res = r
                nbr_h   = height_label[v]
    elif lane == _DIR_DOWN:
        if y + np.int32(1) < H:
            v = n + W
            r = res_down[n]
            if r > np.float32(0.0):
                nbr_idx = v
                nbr_res = r
                nbr_h   = height_label[v]
    elif lane == _DIR_LEFT:
        if x > np.int32(0):
            v = n - np.int32(1)
            r = res_left[n]
            if r > np.float32(0.0):
                nbr_idx = v
                nbr_res = r
                nbr_h   = height_label[v]
    else:  # _DIR_UP
        if y > np.int32(0):
            v = n - W
            r = res_up[n]
            if r > np.float32(0.0):
                nbr_idx = v
                nbr_res = r
                nbr_h   = height_label[v]

    # Also consider sink (encoded as a pseudo-lane-4 entry stored in lane 0 slot
    # after reduction — we handle it separately below for simplicity)
    sh_h  [base] = nbr_h
    sh_nbr[base] = nbr_idx
    sh_res[base] = nbr_res
    cuda.syncwarp()

    # Step 2: lane 0 reduces over the tile to find MHNS
    if lane == 0:
        ecc = excess[n]
        if ecc <= np.float32(1e-7):
            return

        # Parallel reduction for min height
        min_h   = _INF_H
        min_dir = np.int32(-1)
        for d in range(_TILE):
            hd = sh_h[tile * _TILE + d]
            if hd < min_h:
                min_h   = hd
                min_dir = np.int32(d)

        # Also consider the sink arc
        r_snk = res_snk[n]
        if r_snk > np.float32(0.0):
            h_snk = height_label[SINK]
            if h_snk < min_h:
                min_h   = h_snk
                min_dir = np.int32(4)   # special: sink

        if min_h >= _INF_H:
            return   # no admissible or relabel-able neighbour

        # Push condition: hn == min_h + 1  (admissible arc)
        if hn == min_h + np.int32(1):
            if min_dir == np.int32(4):
                delta = ecc if ecc < r_snk else r_snk
                res_snk[n] -= delta
                cuda.atomic.add(excess, SINK, delta)
                cuda.atomic.add(excess, n, -delta)
            else:
                v     = sh_nbr[tile * _TILE + min_dir]
                r_cap = sh_res[tile * _TILE + min_dir]
                delta = ecc if ecc < r_cap else r_cap
                # push from n → v, update reverse residual
                if min_dir == np.int32(_DIR_RIGHT):
                    cuda.atomic.add(res_right, n,  -delta)
                    cuda.atomic.add(res_left,  v,   delta)
                elif min_dir == np.int32(_DIR_DOWN):
                    cuda.atomic.add(res_down, n,  -delta)
                    cuda.atomic.add(res_up,   v,   delta)
                elif min_dir == np.int32(_DIR_LEFT):
                    cuda.atomic.add(res_left,  n,  -delta)
                    cuda.atomic.add(res_right, v,   delta)
                else:  # UP
                    cuda.atomic.add(res_up,   n,  -delta)
                    cuda.atomic.add(res_down, v,   delta)
                cuda.atomic.add(excess, v, delta)
                cuda.atomic.add(excess, n, -delta)
        else:
            # Relabel: hn = min_h + 1
            height_label[n] = min_h + np.int32(1)


# ── preflow kernel ─────────────────────────────────────────────────────────────

@cuda.jit
def _preflow_kernel(cap_src, cap_snk, res_snk, excess, N):
    """Initialise excess and sink residual from source capacities."""
    n = cuda.grid(1)
    if n >= N:
        return
    res_snk[n] = cap_snk[n]
    excess[n]  = cap_src[n]


# ── active-excess check ────────────────────────────────────────────────────────

@cuda.jit
def _reduce_active_excess(excess, height_label, result, N):
    """Largest excess over the nodes that can still push (height < _INF_H).

    Excess stranded at _INF_H is not pushable, so counting it here keeps the
    loop alive for ever — the same reason _scan_active_vertices skips it.
    """
    sh = cuda.shared.array(shape=(256,), dtype=np.float32)
    tid = cuda.threadIdx.x
    n   = cuda.grid(1)
    val = np.float32(0.0)
    if n < N and height_label[n] < _INF_H:
        val = excess[n]
    sh[tid] = val
    cuda.syncthreads()

    stride = cuda.blockDim.x >> 1
    while stride > 0:
        if tid < stride:
            if sh[tid + stride] > sh[tid]:
                sh[tid] = sh[tid + stride]
        cuda.syncthreads()
        stride >>= 1

    if tid == 0:
        cuda.atomic.max(result, 0, sh[0])


# ── CPU global relabeling (BFS from sink on reverse-residual) ─────────────────

@njit(cache=True)
def _global_relabel(res_right, res_down, res_left, res_up, res_snk,
                    H, W, height_label):
    """BFS from SINK on reverse-residual graph to get exact distance labels."""
    N    = H * W
    SINK = N + np.int32(1)

    for i in range(N + 2):
        height_label[i] = _INF_H

    queue = np.empty(N + 2, dtype=np.int32)
    head  = np.int32(0)
    tail  = np.int32(0)

    height_label[SINK] = np.int32(0)
    queue[tail] = SINK
    tail += np.int32(1)

    while head < tail:
        u  = queue[head]
        head += np.int32(1)
        hu = height_label[u]

        if u == SINK:
            # Reverse of sink arcs: any n with res_snk[n]>0 can reach SINK
            for nb in range(N):
                if height_label[nb] == _INF_H and res_snk[nb] > np.float32(0.0):
                    height_label[nb] = hu + np.int32(1)
                    queue[tail] = nb
                    tail += np.int32(1)
        else:
            uy = u // W
            ux = u % W
            # reverse of right(u) = left(u+1) → look at res_left[u]
            if ux + np.int32(1) < W:
                v = u + np.int32(1)
                if height_label[v] == _INF_H and res_left[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)
            # reverse of left(u) = right(u-1) → res_right[u]
            if ux > np.int32(0):
                v = u - np.int32(1)
                if height_label[v] == _INF_H and res_right[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)
            # reverse of down(u) = up(u+W)
            if uy + np.int32(1) < H:
                v = u + W
                if height_label[v] == _INF_H and res_up[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)
            # reverse of up(u) = down(u-W)
            if uy > np.int32(0):
                v = u - W
                if height_label[v] == _INF_H and res_down[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)

    height_label[N] = np.int32(N + 2)   # source — always above everything


# ── public API ─────────────────────────────────────────────────────────────────

def push_relabel_gpu(cap_src, cap_snk, cap_right, cap_down,
                     H, W, max_iter=300, relabel_freq=25):
    """GPU WBPR push-relabel on a 4-connected pixel grid.

    All inputs are 1-D host float32 arrays of length N = H*W.
    Returns a 1-D int32 labeling of length N:
      0  = source side  (foreground)
      1  = sink   side  (background)
    """
    N    = np.int32(H * W)
    SINK = N + np.int32(1)

    # ── initialise residual arrays ─────────────────────────────────────────────
    # Symmetric: res_left = cap_right (reverse of right), res_up = cap_down (reverse of down)
    d_cap_src   = cuda.to_device(cap_src.astype(np.float32))
    d_cap_snk   = cuda.to_device(cap_snk.astype(np.float32))
    res_right = cuda.to_device(cap_right.astype(np.float32))
    res_down  = cuda.to_device(cap_down.astype(np.float32))
    _rev_left, _rev_up = reverse_arcs(cap_right, cap_down, H, W)
    res_left  = cuda.to_device(_rev_left)
    res_up    = cuda.to_device(_rev_up)
    res_snk   = cuda.to_device(np.zeros(int(N), np.float32))
    excess    = cuda.to_device(np.zeros(int(N) + 2, np.float32))

    height_label = np.zeros(int(N) + 2, dtype=np.int32)

    # ── preflow ────────────────────────────────────────────────────────────────
    pf_grid = (int(N) + 255) // 256
    _preflow_kernel[pf_grid, 256](
        d_cap_src, d_cap_snk, res_snk, excess, N)
    cuda.synchronize()

    # Initial global relabeling from host (sets correct BFS labels)
    h_res_right = res_right.copy_to_host()
    h_res_down  = res_down.copy_to_host()
    h_res_left  = res_left.copy_to_host()
    h_res_up    = res_up.copy_to_host()
    h_res_snk   = res_snk.copy_to_host()

    _global_relabel(h_res_right, h_res_down, h_res_left, h_res_up, h_res_snk,
                    H, W, height_label)
    height_label[int(N)] = int(N) + 2   # source

    d_height = cuda.to_device(height_label)

    # ── main loop ──────────────────────────────────────────────────────────────
    # AVQ buffers
    d_avq      = cuda.to_device(np.zeros(int(N), np.int32))
    d_avq_size = cuda.to_device(np.zeros(1, np.int32))
    # Active-excess check
    d_max_exc  = cuda.to_device(np.zeros(1, np.float32))

    scan_block = 256
    exc_block  = 256
    exc_grid   = (int(N) + exc_block - 1) // exc_block

    zero_avq  = np.zeros(1, np.int32)
    zero_exc  = np.zeros(1, np.float32)

    converged = False
    for iteration in range(max_iter):
        # Reset AVQ size
        d_avq_size.copy_to_device(zero_avq)

        # Build AVQ
        scan_grid = (int(N) + scan_block - 1) // scan_block
        _scan_active_vertices[scan_grid, scan_block](
            excess, d_height, d_avq, d_avq_size, N)
        cuda.synchronize()

        avq_size = int(d_avq_size.copy_to_host()[0])
        if avq_size == 0:
            converged = True
            break

        # Tiled push-relabel
        tile_grid = (avq_size + _TILES_PER_BLOCK - 1) // _TILES_PER_BLOCK
        _tiled_push_relabel[tile_grid, _BLOCK](
            d_avq, d_avq_size,
            excess, d_height,
            res_right, res_down, res_left, res_up, res_snk,
            H, W, N)
        cuda.synchronize()

        # Check convergence
        d_max_exc.copy_to_device(zero_exc)
        _reduce_active_excess[exc_grid, exc_block](excess, d_height, d_max_exc, N)
        cuda.synchronize()
        if float(d_max_exc.copy_to_host()[0]) <= 1e-7:
            converged = True
            break

        # Periodic global relabeling (CPU BFS)
        if (iteration + 1) % relabel_freq == 0:
            h_res_right = res_right.copy_to_host()
            h_res_down  = res_down.copy_to_host()
            h_res_left  = res_left.copy_to_host()
            h_res_up    = res_up.copy_to_host()
            h_res_snk   = res_snk.copy_to_host()
            _global_relabel(h_res_right, h_res_down, h_res_left, h_res_up,
                            h_res_snk, H, W, height_label)
            height_label[int(N)] = int(N) + 2
            d_height.copy_to_device(height_label)

    if not converged:
        # Exhausting the cap does not mean the preflow finished; the cut below
        # is then not minimal. Say so rather than return a plausible mask.
        warnings.warn(
            f"push_relabel_gpu hit its {max_iter}-iteration cap at {W}x{H}; "
            "the cut is not minimal.", RuntimeWarning, stacklevel=2)

    # Final global relabeling to extract the cut
    h_res_right = res_right.copy_to_host()
    h_res_down  = res_down.copy_to_host()
    h_res_left  = res_left.copy_to_host()
    h_res_up    = res_up.copy_to_host()
    h_res_snk   = res_snk.copy_to_host()
    _global_relabel(h_res_right, h_res_down, h_res_left, h_res_up,
                    h_res_snk, H, W, height_label)

    labeling = np.zeros(int(N), dtype=np.int32)
    for n in range(int(N)):
        if height_label[n] < _INF_H:
            labeling[n] = np.int32(1)  # sink side = background
    return labeling


def warmup_push_relabel_gpu():
    """JIT-compile all CUDA kernels on tiny dummy data."""
    H, W = np.int32(4), np.int32(4)
    N = H * W
    push_relabel_gpu(
        np.zeros(int(N), np.float32),
        np.zeros(int(N), np.float32),
        np.zeros(int(N), np.float32),
        np.zeros(int(N), np.float32),
        H, W, max_iter=2, relabel_freq=1,
    )
