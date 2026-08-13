import numpy as np
from numba import njit, prange

_INF_HEIGHT = np.int32(2_000_000)


@njit(cache=True)
def _bfs_heights_from_sink(res_right, res_down, res_left, res_up,
                            res_snk, H, W, height_label):
    """BFS from SINK on the reverse-residual graph."""
    N = H * W
    SINK = np.int32(N + 1)

    for i in range(N + 2):
        height_label[i] = _INF_HEIGHT

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
            for n in range(N):
                if height_label[n] == _INF_HEIGHT and res_snk[n] > np.float32(0.0):
                    height_label[n] = hu + np.int32(1)
                    queue[tail] = n
                    tail += np.int32(1)
        else:
            uy = u // W
            ux = u % W
            if ux + np.int32(1) < W:
                v = u + np.int32(1)
                if height_label[v] == _INF_HEIGHT and res_left[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)
            if ux > np.int32(0):
                v = u - np.int32(1)
                if height_label[v] == _INF_HEIGHT and res_right[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)
            if uy + np.int32(1) < H:
                v = u + W
                if height_label[v] == _INF_HEIGHT and res_up[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)
            if uy > np.int32(0):
                v = u - W
                if height_label[v] == _INF_HEIGHT and res_down[v] > np.float32(0.0):
                    height_label[v] = hu + np.int32(1)
                    queue[tail] = v
                    tail += np.int32(1)

    height_label[N] = np.int32(N + 2)


@njit(parallel=True, cache=True)
def _push_color_class(row_par, col_par, H, W,
                      excess, height_label,
                      res_right, res_down, res_left, res_up,
                      res_src, res_snk):
    """Push excess from one checkerboard color class."""
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

            if x + 1 < W:
                v = n + 1
                if res_right[n] > np.float32(0.0) and height_label[v] == hn - 1:
                    delta = excess[n] if excess[n] < res_right[n] else res_right[n]
                    res_right[n] -= delta; res_left[v] += delta
                    excess[n] -= delta;    excess[v] += delta

            if excess[n] <= np.float32(0.0):
                continue

            if x > 0:
                v = n - 1
                if res_left[n] > np.float32(0.0) and height_label[v] == hn - 1:
                    delta = excess[n] if excess[n] < res_left[n] else res_left[n]
                    res_left[n] -= delta;  res_right[v] += delta
                    excess[n] -= delta;    excess[v] += delta

            if excess[n] <= np.float32(0.0):
                continue

            if y + 1 < H:
                v = n + W
                if res_down[n] > np.float32(0.0) and height_label[v] == hn - 1:
                    delta = excess[n] if excess[n] < res_down[n] else res_down[n]
                    res_down[n] -= delta;  res_up[v] += delta
                    excess[n] -= delta;    excess[v] += delta

            if excess[n] <= np.float32(0.0):
                continue

            if y > 0:
                v = n - W
                if res_up[n] > np.float32(0.0) and height_label[v] == hn - 1:
                    delta = excess[n] if excess[n] < res_up[n] else res_up[n]
                    res_up[n] -= delta;   res_down[v] += delta
                    excess[n] -= delta;   excess[v] += delta

            if excess[n] <= np.float32(0.0):
                continue

            if res_snk[n] > np.float32(0.0) and height_label[SINK] == hn - 1:
                delta = excess[n] if excess[n] < res_snk[n] else res_snk[n]
                res_snk[n] -= delta; excess[SINK] += delta; excess[n] -= delta


@njit(parallel=True, cache=True)
def _relabel_pass(H, W, excess, height_label,
                  res_right, res_down, res_left, res_up,
                  res_src, res_snk):
    """Relabel all active nodes in parallel."""
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

            new_h = _INF_HEIGHT
            if x + 1 < W and res_right[n] > np.float32(0.0):
                h_n = height_label[n + 1] + np.int32(1)
                if h_n < new_h: new_h = h_n
            if x > 0 and res_left[n] > np.float32(0.0):
                h_n = height_label[n - 1] + np.int32(1)
                if h_n < new_h: new_h = h_n
            if y + 1 < H and res_down[n] > np.float32(0.0):
                h_n = height_label[n + W] + np.int32(1)
                if h_n < new_h: new_h = h_n
            if y > 0 and res_up[n] > np.float32(0.0):
                h_n = height_label[n - W] + np.int32(1)
                if h_n < new_h: new_h = h_n
            if res_snk[n] > np.float32(0.0):
                h_n = height_label[SINK] + np.int32(1)
                if h_n < new_h: new_h = h_n
            if new_h < _INF_HEIGHT:
                height_label[n] = new_h


@njit(cache=True)
def push_relabel(cap_src, cap_snk, cap_right, cap_down,
                 H, W, max_iter=200, relabel_freq=20):
    """FIFO-style parallel push-relabel max-flow on a 4-connected pixel grid."""
    N = np.int32(H * W)
    SOURCE = N
    SINK = N + np.int32(1)

    res_src   = cap_src.copy()
    res_snk   = cap_snk.copy()
    res_right = cap_right.copy()
    res_down  = cap_down.copy()
    # res_left[n] is the arc n -> n-1 and its capacity is cap_right[n-1], not
    # cap_right[n]; res_up[n] pairs with cap_down[n-W]. Copying cap_right and
    # cap_down straight across shifts every reverse arc one pixel and wraps a
    # row boundary at x == 0, which leaves the residual graph inconsistent and
    # the cut above the minimum — measured 4 of 4 random grids, up to 3.449%.
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

    _bfs_heights_from_sink(res_right, res_down, res_left, res_up,
                           res_snk, H, W, height_label)

    total_src = np.float32(0.0)
    for n in range(N):
        e = cap_src[n]
        excess[n] = e
        res_src[n] = np.float32(0.0)
        total_src += e
    excess[SOURCE] = -total_src

    for iteration in range(max_iter):
        _push_color_class(0, 0, H, W, excess, height_label,
                          res_right, res_down, res_left, res_up, res_src, res_snk)
        _push_color_class(0, 1, H, W, excess, height_label,
                          res_right, res_down, res_left, res_up, res_src, res_snk)
        _push_color_class(1, 0, H, W, excess, height_label,
                          res_right, res_down, res_left, res_up, res_src, res_snk)
        _push_color_class(1, 1, H, W, excess, height_label,
                          res_right, res_down, res_left, res_up, res_src, res_snk)

        _relabel_pass(H, W, excess, height_label,
                      res_right, res_down, res_left, res_up, res_src, res_snk)

        if iteration % relabel_freq == 0:
            _bfs_heights_from_sink(res_right, res_down, res_left, res_up,
                                   res_snk, H, W, height_label)

        active = False
        for n in range(N):
            if excess[n] > np.float32(1e-6):
                active = True
                break
        if not active:
            break

    _bfs_heights_from_sink(res_right, res_down, res_left, res_up,
                           res_snk, H, W, height_label)

    labeling = np.zeros(N, dtype=np.int32)
    for n in range(N):
        if height_label[n] < _INF_HEIGHT:
            labeling[n] = np.int32(1)
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
