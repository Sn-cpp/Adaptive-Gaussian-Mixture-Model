"""Push-Relabel must return a genuine minimum cut, not merely a cut.

A wrong max-flow is invisible: the mask still looks like a mask. The only way
to know is to compare against an independent solver, so every test here scores
our cut against Boykov-Kolmogorov via PyMaxflow.

    pip install PyMaxflow
    python -m pytest test_push_relabel.py -q
"""
import numpy as np
import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from graphcut.push_relabel_numba import push_relabel

maxflow = pytest.importorskip("maxflow", reason="PyMaxflow not installed")


def reference_mincut(cap_src, cap_snk, cap_right, cap_down, H, W):
    """Boykov-Kolmogorov reference. Returns (flow, segment) with 0=SOURCE side."""
    g = maxflow.Graph[float]()
    nodes = g.add_nodes(H * W)
    for n in range(H * W):
        g.add_tedge(nodes[n], float(cap_src[n]), float(cap_snk[n]))
    for y in range(H):
        for x in range(W):
            n = y * W + x
            if x + 1 < W and cap_right[n] > 0:
                g.add_edge(nodes[n], nodes[n + 1],
                           float(cap_right[n]), float(cap_right[n]))
            if y + 1 < H and cap_down[n] > 0:
                g.add_edge(nodes[n], nodes[n + W],
                           float(cap_down[n]), float(cap_down[n]))
    flow = g.maxflow()
    seg = np.array([g.get_segment(nodes[n]) for n in range(H * W)], np.int32)
    return flow, seg


def cut_value(labeling, cap_src, cap_snk, cap_right, cap_down, H, W):
    """Total weight of edges crossing from the SOURCE side to the SINK side.

    labeling: 0 = SOURCE side, 1 = SINK side. A SOURCE->pixel edge is cut when
    the pixel sits on the SINK side, and vice versa.
    """
    total = float(np.where(labeling == 1, cap_src, cap_snk).sum())
    lab2d = labeling.reshape(H, W)
    r = cap_right.reshape(H, W)[:, :-1]
    total += float(r[lab2d[:, :-1] != lab2d[:, 1:]].sum())
    d = cap_down.reshape(H, W)[:-1, :]
    total += float(d[lab2d[:-1, :] != lab2d[1:, :]].sum())
    return total


def random_grid(H, W, seed):
    rng = np.random.default_rng(seed)
    N = H * W
    cap_src = (rng.random(N) * 10).astype(np.float32)
    cap_snk = (rng.random(N) * 10).astype(np.float32)
    cap_right = (rng.random(N) * 3).astype(np.float32)
    cap_down = (rng.random(N) * 3).astype(np.float32)
    for y in range(H):
        cap_right[y * W + W - 1] = 0.0        # no edge off the right border
    cap_down[(H - 1) * W:] = 0.0              # none off the bottom border
    return cap_src, cap_snk, cap_right, cap_down


@pytest.mark.parametrize("seed", range(8))
def test_cut_is_minimum(seed):
    """The cut we return must weigh exactly the reference max-flow.

    Regression guard for the reverse-arc indexing bug: res_left[n] is the arc
    n -> n-1 and pairs with cap_right[n-1], but the initialiser copied
    cap_right straight across, shifting every reverse arc one pixel. Every cut
    came out above the minimum -- 12/12 random grids, by 0.02% to 4% -- while
    still looking like a plausible segmentation.
    """
    rng = np.random.default_rng(1000 + seed)
    H, W = int(rng.integers(4, 13)), int(rng.integers(4, 13))
    caps = random_grid(H, W, seed)

    flow_ref, seg_ref = reference_mincut(*caps, H, W)
    labeling = push_relabel(*caps, np.int32(H), np.int32(W),
                            max_iter=20000, relabel_freq=10)

    ours = cut_value(labeling, *caps, H, W)
    assert ours == pytest.approx(flow_ref, rel=1e-4, abs=1e-3), (
        f"{H}x{W}: cut {ours:.4f} against reference max-flow {flow_ref:.4f}")
    assert np.array_equal(labeling, seg_ref), (
        f"{H}x{W}: {int((labeling != seg_ref).sum())} pixels labelled differently")


def test_terminates_when_excess_is_stranded():
    """Excess parked on a node that cannot reach the SINK must not keep the
    loop alive.

    _push_color_class skips nodes at _INF_HEIGHT, so if the termination check
    counts them as active the loop can only end by exhausting max_iter. That
    made the runtime swing wildly with the data -- 9.3 s at 60x80 against
    0.24 s at 120x160 -- so this pins the fix with a generous time budget and a
    grid built to strand flow: a well of source capacity walled off from the
    sink by zero-capacity n-links.
    """
    import time
    H = W = 40
    N = H * W
    cap_src = np.zeros(N, np.float32)
    cap_snk = np.zeros(N, np.float32)
    cap_right = np.zeros(N, np.float32)
    cap_down = np.zeros(N, np.float32)

    for y in range(H):
        for x in range(W):
            n = y * W + x
            if y < H // 2:
                cap_src[n] = 5.0            # top half is pumped from SOURCE
            else:
                cap_snk[n] = 5.0            # bottom half drains to SINK
            # n-links only *within* each half: nothing crosses the midline,
            # so the top half's excess can never reach the SINK.
            if x + 1 < W:
                cap_right[n] = 2.0
            if y + 1 < H and y + 1 != H // 2:
                cap_down[n] = 2.0

    push_relabel(cap_src, cap_snk, cap_right, cap_down,
                 np.int32(4), np.int32(4), max_iter=2, relabel_freq=1)  # JIT

    start = time.perf_counter()
    labeling = push_relabel(cap_src, cap_snk, cap_right, cap_down,
                            np.int32(H), np.int32(W),
                            max_iter=20000, relabel_freq=20)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"took {elapsed:.1f}s — the loop is spinning to max_iter"
    flow_ref, seg_ref = reference_mincut(cap_src, cap_snk, cap_right, cap_down, H, W)
    assert np.array_equal(labeling, seg_ref)


def test_all_source_side_when_sink_is_unreachable():
    """Degenerate graph: no pixel has a SINK edge, so nothing can be cut off."""
    H = W = 8
    N = H * W
    cap_src = np.full(N, 3.0, np.float32)
    cap_snk = np.zeros(N, np.float32)
    cap_right = np.zeros(N, np.float32)
    cap_down = np.zeros(N, np.float32)

    labeling = push_relabel(cap_src, cap_snk, cap_right, cap_down,
                            np.int32(H), np.int32(W), max_iter=500, relabel_freq=20)

    assert (labeling == 0).all(), "with no path to the SINK every node is SOURCE side"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
