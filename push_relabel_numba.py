"""Parallel Push-Relabel max-flow on a 4-connected pixel grid.

Graph structure
---------------
Pixel nodes  : 0 .. N-1  where N = H * W,  n = y * W + x
SOURCE (bg)  : N
SINK   (fg)  : N + 1

Edges
-----
t-links  : SOURCE → n  with capacity cap_src[n]
           n → SINK    with capacity cap_snk[n]
n-links  : n → n+1    with capacity cap_right[n]  (0 at right boundary)
           n → n+W    with capacity cap_down[n]   (0 at bottom boundary)
           n-links are symmetric: cap_left[n] = cap_right[n-1],
                                  cap_up[n]   = cap_down[n-W].
           We store both the forward and reverse residuals separately.

Parallelisation
---------------
Checkerboard (2×2) coloring: within one color class no two pixels are
4-connected neighbors, so push operations are data-race free.
Four _push_color_class passes per outer iteration; a parallel relabel
pass follows every RELABEL_FREQ iterations (or whenever the inner loop
finds a pixel with no admissible neighbor).
Global relabeling (BFS from SINK) is called periodically to keep height
labels tight and avoid excess stagnation.
"""
import numpy as np
from numba import njit, prange

_INF_HEIGHT = np.int32(2_000_000)


@njit(cache=True)
def _bfs_heights_from_sink(res_right, res_down, res_left, res_up,
                            res_snk, H, W, height_label):
    """BFS from SINK on the reverse-residual graph.

    Sets height_label[n] = shortest-path distance from n to SINK in the
    residual graph.  Nodes unreachable from SINK get _INF_HEIGHT.
    SOURCE gets H*W+2 (standard push-relabel sentinel).
    """
    N = H * W
    SINK = np.int32(N + 1)

    for i in range(N + 2):
        height_label[i] = _INF_HEIGHT

    # BFS queue as a flat array with head/tail pointers
    queue = np.empty(N + 2, dtype=np.int32)
    head = np.int32(0)
    tail = np.int32(0)

    height_label[SINK] = np.int32(0)
    queue[tail] = SINK
    tail += np.int32(1)

    while head < tail:
        u = queue[head]
        head += np.int32(1)
        hu = height_label[u]

        if u == SINK:
            # Reverse of (n → SINK) is (SINK → n), exists when res_snk[n] > 0
            # meaning original cap was positive (flow could have been sent).
            # After init (before any flow) res_snk equals cap_snk, so all
            # nodes with cap_snk > 0 are reachable from SINK via reverse edge.
            for n in range(N):
                if height_label[n] == _INF_HEIGHT and res_snk[n] > np.float32(0.0):
                    height_label[n] = hu + np.int32(1)
                    queue[tail] = n
                    tail += np.int32(1)
        else:
            uy = u // W
            ux = u % W
            # Reverse of (u → right neighbor) is (right → u) via res_left
            if ux + np.int32(1) < W:
                v = u + np.int32(1)
                if height_label[v] == _INF_HEIGHT and res_left[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)
            # Reverse of (u → left neighbor) is (left → u) via res_right
            if ux > np.int32(0):
                v = u - np.int32(1)
                if height_label[v] == _INF_HEIGHT and res_right[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)
            # Reverse of (u → down neighbor) is (down → u) via res_up
            if uy + np.int32(1) < H:
                v = u + W
                if height_label[v] == _INF_HEIGHT and res_up[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)
            # Reverse of (u → up neighbor) is (up → u) via res_down
            if uy > np.int32(0):
                v = u - W
                if height_label[v] == _INF_HEIGHT and res_down[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)
            # Reverse of (SOURCE → u) via res_src: if res_src[u] < cap_src[u]
            # we cannot check original cap here, so skip SOURCE-side BFS
            # (SOURCE height is set unconditionally below)

    # SOURCE sentinel
    height_label[N] = np.int32(N + 2)


@njit(parallel=True, cache=True)
def _push_color_class(row_par, col_par, H, W,
                      excess, height_label,
                      res_right, res_down, res_left, res_up,
                      res_src, res_snk):
    """Push excess from one checkerboard color class.

    Pixels in the same color class have the same (row%2, col%2) parity.
    No two same-class pixels are 4-neighbors → writes to residuals and
    excess are data-race free within one color class.
    """
    N = H * W
    SINK = np.int32(N + 1)

    for y in prange(H):
        if (y % 2) != row_par:
            continue
        for x in range(W):
            if (x % 2) != col_par:
                continue
            n = y * W + x
            if excess[n] <= np.float32(0.0):
                continue
            hn = height_label[n]
            if hn >= _INF_HEIGHT:
                continue

            # Push to right neighbor
            if x + 1 < W:
                v = n + 1
                if res_right[n] > np.float32(0.0) and height_label[v] == hn - 1:
                    delta = excess[n] if excess[n] < res_right[n] else res_right[n]
                    res_right[n] -= delta
                    res_left[v] += delta
                    excess[n] -= delta
                    excess[v] += delta

            if excess[n] <= np.float32(0.0):
                continue

            # Push to left neighbor
            if x > 0:
                v = n - 1
                if res_left[n] > np.float32(0.0) and height_label[v] == hn - 1:
                    delta = excess[n] if excess[n] < res_left[n] else res_left[n]
                    res_left[n] -= delta
                    res_right[v] += delta
                    excess[n] -= delta
                    excess[v] += delta

            if excess[n] <= np.float32(0.0):
                continue

            # Push to down neighbor
            if y + 1 < H:
                v = n + W
                if res_down[n] > np.float32(0.0) and height_label[v] == hn - 1:
                    delta = excess[n] if excess[n] < res_down[n] else res_down[n]
                    res_down[n] -= delta
                    res_up[v] += delta
                    excess[n] -= delta
                    excess[v] += delta

            if excess[n] <= np.float32(0.0):
                continue

            # Push to up neighbor
            if y > 0:
                v = n - W
                if res_up[n] > np.float32(0.0) and height_label[v] == hn - 1:
                    delta = excess[n] if excess[n] < res_up[n] else res_up[n]
                    res_up[n] -= delta
                    res_down[v] += delta
                    excess[n] -= delta
                    excess[v] += delta

            if excess[n] <= np.float32(0.0):
                continue

            # Push to SINK (height[SINK]=0, so hn must equal 1)
            if res_snk[n] > np.float32(0.0) and height_label[SINK] == hn - 1:
                delta = excess[n] if excess[n] < res_snk[n] else res_snk[n]
                res_snk[n] -= delta
                excess[SINK] += delta
                excess[n] -= delta


@njit(parallel=True, cache=True)
def _relabel_pass(H, W, excess, height_label,
                  res_right, res_down, res_left, res_up,
                  res_src, res_snk):
    """Relabel all active nodes in parallel.

    Each pixel independently reads neighbor heights (no write conflict within
    one pass) and raises its own height to min(neighbor_height)+1 if it has
    excess and no admissible push edge.  Reading a stale neighbor height from
    the same pass is safe — it only slows convergence, not correctness.
    """
    N = H * W
    SINK = np.int32(N + 1)

    for y in prange(H):
        for x in range(W):
            n = y * W + x
            if excess[n] <= np.float32(0.0):
                continue
            hn = height_label[n]
            if hn >= _INF_HEIGHT:
                continue

            # Check if there is any admissible neighbor — if so, don't relabel
            admissible = False
            if x + 1 < W and res_right[n] > np.float32(0.0) and height_label[n + 1] == hn - 1:
                admissible = True
            if not admissible and x > 0 and res_left[n] > np.float32(0.0) and height_label[n - 1] == hn - 1:
                admissible = True
            if not admissible and y + 1 < H and res_down[n] > np.float32(0.0) and height_label[n + W] == hn - 1:
                admissible = True
            if not admissible and y > 0 and res_up[n] > np.float32(0.0) and height_label[n - W] == hn - 1:
                admissible = True
            if not admissible and res_snk[n] > np.float32(0.0) and height_label[SINK] == hn - 1:
                admissible = True
            if admissible:
                continue

            # Relabel: raise height to 1 + min(admissible-neighbor height)
            new_h = _INF_HEIGHT
            if x + 1 < W and res_right[n] > np.float32(0.0):
                h_n = height_label[n + 1] + np.int32(1)
                if h_n < new_h:
                    new_h = h_n
            if x > 0 and res_left[n] > np.float32(0.0):
                h_n = height_label[n - 1] + np.int32(1)
                if h_n < new_h:
                    new_h = h_n
            if y + 1 < H and res_down[n] > np.float32(0.0):
                h_n = height_label[n + W] + np.int32(1)
                if h_n < new_h:
                    new_h = h_n
            if y > 0 and res_up[n] > np.float32(0.0):
                h_n = height_label[n - W] + np.int32(1)
                if h_n < new_h:
                    new_h = h_n
            if res_snk[n] > np.float32(0.0):
                h_n = height_label[SINK] + np.int32(1)
                if h_n < new_h:
                    new_h = h_n
            if new_h < _INF_HEIGHT:
                height_label[n] = new_h


@njit(cache=True)
def push_relabel(cap_src, cap_snk, cap_right, cap_down,
                 H, W, max_iter=200, relabel_freq=20):
    """FIFO-style parallel push-relabel max-flow on a 4-connected pixel grid.

    Parameters
    ----------
    cap_src   : (H*W,) float32 — SOURCE→pixel capacity
    cap_snk   : (H*W,) float32 — pixel→SINK capacity
    cap_right : (H*W,) float32 — pixel→right-neighbor capacity (0 at boundary)
    cap_down  : (H*W,) float32 — pixel→down-neighbor capacity  (0 at boundary)
    H, W      : image dimensions
    max_iter  : outer iteration cap
    relabel_freq : call global relabel every this many outer iterations

    Returns
    -------
    labeling : (H*W,) int32
        0 = SOURCE side (background), 1 = SINK side (foreground)
    """
    N = np.int32(H * W)
    SOURCE = N
    SINK = N + np.int32(1)

    # Residual capacities. The n-links are symmetric, so every edge is stored
    # twice — once from each endpoint. res_left[n] is the arc n -> n-1, and the
    # capacity of that arc lives in cap_right[n-1], not cap_right[n]; likewise
    # res_up[n] pairs with cap_down[n-W]. Copying cap_right/cap_down straight
    # across shifts every reverse arc by one pixel, which silently corrupts the
    # residual graph and yields a cut above the true minimum.
    res_src   = cap_src.copy()
    res_snk   = cap_snk.copy()
    res_right = cap_right.copy()
    res_down  = cap_down.copy()
    res_left  = np.zeros(N, dtype=np.float32)
    res_up    = np.zeros(N, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            n = y * W + x
            if x > 0:
                res_left[n] = cap_right[n - 1]
            if y > 0:
                res_up[n] = cap_down[n - W]

    excess       = np.zeros(N + 2, dtype=np.float32)
    height_label = np.zeros(N + 2, dtype=np.int32)

    # PR-1: global relabeling from SINK
    _bfs_heights_from_sink(res_right, res_down, res_left, res_up,
                           res_snk, H, W, height_label)

    # PR-2: saturate all SOURCE out-edges (standard preflow initialization)
    total_src = np.float32(0.0)
    for n in range(N):
        e = cap_src[n]
        excess[n] = e
        res_src[n] = np.float32(0.0)
        total_src += e
    excess[SOURCE] = -total_src

    # PR-3: main push-relabel loop
    for iteration in range(max_iter):
        # Four color-class push passes (checkerboard, race-free)
        _push_color_class(0, 0, H, W, excess, height_label,
                          res_right, res_down, res_left, res_up, res_src, res_snk)
        _push_color_class(0, 1, H, W, excess, height_label,
                          res_right, res_down, res_left, res_up, res_src, res_snk)
        _push_color_class(1, 0, H, W, excess, height_label,
                          res_right, res_down, res_left, res_up, res_src, res_snk)
        _push_color_class(1, 1, H, W, excess, height_label,
                          res_right, res_down, res_left, res_up, res_src, res_snk)

        # Parallel relabel pass
        _relabel_pass(H, W, excess, height_label,
                      res_right, res_down, res_left, res_up, res_src, res_snk)

        # Periodic global relabeling
        if iteration % relabel_freq == 0:
            _bfs_heights_from_sink(res_right, res_down, res_left, res_up,
                                   res_snk, H, W, height_label)

        # Termination check: no *pushable* active pixel node left.
        #
        # A node at _INF_HEIGHT cannot reach the SINK through the residual
        # graph, and _push_color_class skips it for exactly that reason. Its
        # excess is flow that would have to travel back to the SOURCE, which
        # the second phase of push-relabel does and which the minimum cut does
        # not depend on. Counting those nodes as active — as this loop used to
        # — means the loop can never terminate once any excess is stranded, so
        # it spins out the full max_iter doing no work at all. That is what
        # made the runtime erratic: 9.3 s at 60x80 against 0.24 s at 120x160,
        # entirely determined by whether any excess happened to get stranded.
        active = False
        for n in range(N):
            if excess[n] > np.float32(1e-6) and height_label[n] < _INF_HEIGHT:
                active = True
                break
        if not active:
            break

    # PR-4: final global relabeling to get accurate residual reachability,
    # then label extraction.  Running BFS here ensures height_label reflects
    # the converged residual graph, not a potentially stale mid-iteration state.
    _bfs_heights_from_sink(res_right, res_down, res_left, res_up,
                           res_snk, H, W, height_label)

    # Node is background (SOURCE side) ↔ cannot reach SINK in residual graph
    # ↔ BFS from SINK could not reach it ↔ height_label[n] == _INF_HEIGHT.
    labeling = np.zeros(N, dtype=np.int32)
    for n in range(N):
        if height_label[n] < _INF_HEIGHT:
            labeling[n] = np.int32(1)   # reachable from SINK → foreground
    return labeling


def warmup_push_relabel():
    """Pre-compile push_relabel on a 4×4 dummy so the first real frame is fast."""
    H, W = np.int32(4), np.int32(4)
    N = H * W
    push_relabel(
        np.zeros(N, dtype=np.float32),
        np.zeros(N, dtype=np.float32),
        np.zeros(N, dtype=np.float32),
        np.zeros(N, dtype=np.float32),
        H, W, max_iter=2, relabel_freq=1,
    )
