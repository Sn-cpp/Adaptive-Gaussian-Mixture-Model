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
    labeling, _ = push_relabel(*caps, np.int32(H), np.int32(W),
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
    labeling, _ = push_relabel(cap_src, cap_snk, cap_right, cap_down,
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

    labeling, _ = push_relabel(cap_src, cap_snk, cap_right, cap_down,
                            np.int32(H), np.int32(W), max_iter=500, relabel_freq=20)

    assert (labeling == 0).all(), "with no path to the SINK every node is SOURCE side"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


def test_pipeline_refuses_a_backend_without_bg_prob():
    """A backend that leaves bg_prob at zero must be rejected, not tolerated.

    The GC seed is thresholded out of bg_prob, so an all-zero confidence map
    marks every pixel probable-foreground and the cut segments noise. Silent
    garbage is the worst failure mode here because the output still looks like
    a mask.
    """
    import numpy as np
    from graphcut import GrabCutPipeline
    from gmm.mog2_common import MOG2Base

    class Bogus(MOG2Base):
        FILLS_BG_PROB = False

        def _step_kernel(self, frame, args):
            pass

    frame = np.zeros((16, 16, 3), np.uint8)
    with pytest.raises(ValueError, match="bg_prob"):
        GrabCutPipeline(Bogus(frame, n_components=5), (0, 0, 16, 16))


def test_reaches_the_minimum_at_realistic_size():
    """A 200-iteration cap was fine on tiny grids and wrong where it runs.

    Convergence costs roughly one outer iteration per pixel of image diameter:
    119 at 60x80 but 783 at 240x320. The fixed cap of 200 therefore returned a
    labelling that disagreed with BK on *every* pixel at 240x320, with nothing
    to indicate it. This pins both the answer and the reporting.
    """
    import cv2
    H, W = 120, 160
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (9, 9), 0)
    prob = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (31, 31), 0)
    cap_src = (-np.log(np.clip(1 - prob, 1e-3, 1))).astype(np.float32).ravel() * 5
    cap_snk = (-np.log(np.clip(prob, 1e-3, 1))).astype(np.float32).ravel() * 5
    gx = np.zeros((H, W), np.float32); gx[:, :-1] = np.abs(np.diff(img, axis=1))
    gy = np.zeros((H, W), np.float32); gy[:-1, :] = np.abs(np.diff(img, axis=0))
    beta = 1.0 / (2 * max(float((gx ** 2).mean() + (gy ** 2).mean()), 1e-8))
    cap_right = (50 * np.exp(-beta * gx ** 2)).astype(np.float32); cap_right[:, -1] = 0
    cap_down = (50 * np.exp(-beta * gy ** 2)).astype(np.float32); cap_down[-1, :] = 0
    caps = (cap_src, cap_snk, cap_right.ravel(), cap_down.ravel())

    flow_ref, seg_ref = reference_mincut(*caps, H, W)
    labeling, iterations = push_relabel(*caps, np.int32(H), np.int32(W),
                                        max_iter=20000, relabel_freq=20)

    assert iterations < 20000, "did not converge inside the cap"
    assert np.array_equal(labeling, seg_ref)
    assert cut_value(labeling, *caps, H, W) == pytest.approx(flow_ref, rel=1e-6)


def test_hitting_the_cap_is_reported():
    """Exhausting max_iter must be visible, not silent."""
    H = W = 60
    caps = random_grid(H, W, 42)
    _, iterations = push_relabel(*caps, np.int32(H), np.int32(W),
                                 max_iter=2, relabel_freq=1)
    assert iterations == 2, "a run that used its whole budget must say so"


class TestMorphology:
    """graphcut/morph_numba against OpenCV, on the cases that used to break."""

    def _ref_largest(self, mask):
        import cv2
        n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 4)
        if n <= 1:
            return np.zeros_like(mask)
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return ((lab == k).astype(np.uint8)) * 255

    def test_largest_component_on_an_empty_mask(self):
        """It used to return a fully white frame: with no component found
        best_lbl stayed 0, and 0 is also 'unvisited', so everything matched."""
        from graphcut.morph_numba import largest_component
        empty = np.zeros((8, 8), np.uint8)
        assert not largest_component(empty, 8, 8).any()

    def test_largest_component_accepts_any_nonzero_foreground(self):
        """Seeding tested != 0 while growing tested == 255, so a mask carrying
        any other non-zero value shattered into single-pixel components."""
        from graphcut.morph_numba import largest_component
        mask = np.ones((8, 8), np.uint8)          # non-zero, not 255
        assert int((largest_component(mask, 8, 8) > 0).sum()) == 64

    def test_largest_component_matches_opencv(self):
        from graphcut.morph_numba import largest_component
        mask = np.zeros((10, 10), np.uint8)
        mask[1:4, 1:4] = 255                      # 9 px
        mask[6:8, 6:8] = 255                      # 4 px
        ours = largest_component(mask, 10, 10)
        assert np.array_equal(ours, self._ref_largest(mask))

    def test_close_does_not_eat_the_frame_border(self):
        """_erode counted out-of-bounds as background, so closing shaved a
        `radius`-wide strip off every edge — a subject touching the edge of a
        webcam frame lost it on every frame."""
        from graphcut.morph_numba import morphological_close
        mask = np.zeros((16, 16), np.uint8)
        mask[8:16, 4:12] = 255                    # touches the bottom edge
        tmp, out = np.zeros_like(mask), np.zeros_like(mask)
        morphological_close(mask, tmp, out, 16, 16, radius=2)
        lost = int(((mask > 0) & (out == 0)).sum())
        assert lost == 0, f"closing removed {lost} foreground pixels"


def test_tlink_capacities_are_never_negative():
    """The colour GMM density is unnormalised, so a tight component returns a
    value above 1 and -log of it is negative. Max-flow is undefined for
    negative capacities and push-relabel then returns a non-minimal cut."""
    import cv2
    from gmm import GMM_CPU_NUMBA
    from graphcut import GrabCutPipeline

    rng = np.random.default_rng(0)
    frames = [np.clip(np.full((24, 32, 3), 120, np.int16)
                      + rng.integers(-30, 31, (24, 32, 3)), 0, 255).astype(np.uint8)
              for _ in range(6)]
    for f in frames[2:5]:
        f[6:16, 8:24] = 240

    pipe = GrabCutPipeline(GMM_CPU_NUMBA(frames[0], n_components=5), (0, 0, 32, 24))
    for f in frames:
        pipe.rqstep(f)

    assert (pipe._cap_src >= 0).all(), f"min cap_src {pipe._cap_src.min()}"
    assert (pipe._cap_snk >= 0).all(), f"min cap_snk {pipe._cap_snk.min()}"


def test_colour_gmm_is_deterministic():
    """cv2.kmeans with KMEANS_PP_CENTERS draws from OpenCV's global RNG, so the
    same frame segmented differently every run and nothing was reproducible."""
    from graphcut.fgd_gmm_numba import GrabCutGMM
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (40, 50, 3)).astype(np.float64)
    gc = np.full((40, 50), 2, np.uint8)
    gc[10:30, 10:40] = 3

    runs = []
    for _ in range(3):
        g = GrabCutGMM()
        g.fit(img, gc, is_fg=True)
        runs.append(g.model.copy())
    for r in runs[1:]:
        assert np.array_equal(runs[0], r), "fit is not reproducible"
