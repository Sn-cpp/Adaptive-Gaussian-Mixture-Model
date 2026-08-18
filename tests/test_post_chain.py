"""The GPU post-processing chain must equal the host chain, pixel for pixel.

Run the GPU half on a machine with a device, or anywhere with::

    NUMBA_ENABLE_CUDASIM=1 pytest tests/test_post_chain.py

The host chain in `utils.post_processing` is the specification: it is what
`eval_highway.py` scored, so it is the thing whose F1 we quote. A CUDA kernel
that merely looks similar is not evidence of anything.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2
import numpy as np
import pytest

from settings import MOG2_BG_PROB_THRESHOLD
from utils.post_processing import fill_holes, refine_mask, threshold_bg_prob

H, W = 48, 64


def gpu_available():
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        return True
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not gpu_available(),
    reason="no CUDA device; rerun with NUMBA_ENABLE_CUDASIM=1")


def blobby(seed=0, shape=(H, W)):
    """A mask with the shape real MOG2 output has: blobs, speckle, holes."""
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (9, 9), 0)
    return (field > 0.52).astype(np.uint8) * 255


# ── host chain properties ─────────────────────────────────────────────────────

def test_binary_median_is_a_majority_vote():
    """The identity every CUDA median kernel here relies on.

    A 5x5 median of values drawn from {0, 255} is 255 exactly when at least 13
    of the 25 are — so the kernels count instead of sorting. If OpenCV ever
    stopped agreeing, the kernels would be silently wrong and nothing else
    would catch it.
    """
    for seed in range(5):
        m = blobby(seed)
        opencv = cv2.medianBlur(m, 5)
        pad = cv2.copyMakeBorder(m, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
        counted = np.zeros_like(m, dtype=np.int32)
        for dy in range(5):
            for dx in range(5):
                counted += (pad[dy:dy + H, dx:dx + W] == 255)
        majority = np.where(counted >= 13, np.uint8(255), np.uint8(0))
        assert np.array_equal(opencv, majority)


def test_fill_holes_fills_an_enclosed_gap_without_growing_the_shape():
    m = np.zeros((H, W), np.uint8)
    m[10:38, 12:52] = 255
    m[22:26, 28:34] = 0
    out = fill_holes(m)
    assert out[24, 31] == 255
    outside = np.ones((H, W), bool)
    outside[10:38, 12:52] = False
    assert not out[outside].any(), "fill_holes inflated the mask"


def test_fill_holes_survives_an_object_in_the_top_left_corner():
    """Seeding the flood at (0, 0) instead of a padded border fails here: the
    flood cannot start, nothing is reachable, and every pixel comes back
    foreground."""
    m = np.zeros((H, W), np.uint8)
    m[0:20, 0:20] = 255
    out = fill_holes(m)
    assert out[40, 60] == 0, "the whole frame was declared foreground"
    assert np.array_equal(out, m)


def test_threshold_is_stricter_than_mog2s_own_decision():
    """MOG2 calls a pixel background on any match at all — `bg_prob > 0`.
    Asking for half the weight can only ever mark more pixels foreground."""
    rng = np.random.default_rng(1)
    bg_prob = rng.random((H, W)).astype(np.float32)
    mog2 = np.where(bg_prob <= 0, np.uint8(255), np.uint8(0))
    ours = threshold_bg_prob(bg_prob)
    assert (ours[mog2 == 255] == 255).all()
    assert float(MOG2_BG_PROB_THRESHOLD) > 0.0


# ── GPU kernels against the host chain ────────────────────────────────────────

@requires_gpu
def test_threshold_kernel_matches_numpy():
    from numba import cuda
    from gmm_mask.gpu import post_kernels as pk

    rng = np.random.default_rng(2)
    bg_prob = rng.random((H, W)).astype(np.float32)
    d_out = cuda.device_array((H, W), np.uint8)
    pk.threshold_kernel[pk.grid_for(H, W), (pk.TILE_X, pk.TILE_Y)](
        cuda.to_device(bg_prob), d_out, MOG2_BG_PROB_THRESHOLD)
    assert np.array_equal(d_out.copy_to_host(), threshold_bg_prob(bg_prob))


@requires_gpu
@pytest.mark.parametrize("kernel_name", ["median5_kernel", "median5_tiled_kernel"])
def test_median_kernels_match_opencv(kernel_name):
    """Both the plain and the tiled median must be bit-exact with cv2, and the
    tiled one exists only as an optimisation — if it disagrees, it is a bug in
    the halo, which is the one place a tiled stencil goes wrong."""
    from numba import cuda
    from gmm_mask.gpu import post_kernels as pk

    kernel = getattr(pk, kernel_name)
    for seed in range(3):
        m = blobby(seed)
        d_out = cuda.device_array((H, W), np.uint8)
        kernel[pk.grid_for(H, W), (pk.TILE_X, pk.TILE_Y)](cuda.to_device(m), d_out)
        assert np.array_equal(d_out.copy_to_host(), cv2.medianBlur(m, 5)), \
            f"{kernel_name} disagrees with cv2.medianBlur on seed {seed}"


@requires_gpu
def test_v1_and_v2_produce_the_same_refined_mask_as_the_host_chain():
    """End to end: model + GPU post + host fill == the scored host chain."""
    from gmm_mask import GMM_Mask_CUDA_v1, GMM_Mask_CUDA_v2
    if GMM_Mask_CUDA_v1 is None:
        pytest.skip("CUDA backends unavailable")

    rng = np.random.default_rng(3)
    base = rng.integers(40, 90, (H, W, 3), dtype=np.uint8)
    frames = []
    for i in range(8):
        f = base.astype(np.float32)
        f[12:30, 16:44] = 200.0 if i >= 3 else f[12:30, 16:44]
        frames.append(np.ascontiguousarray(
            np.clip(f + rng.normal(0, 2.0, f.shape), 0, 255), dtype=np.float32))

    outs = {}
    for name, cls in (("v1", GMM_Mask_CUDA_v1), ("v2", GMM_Mask_CUDA_v2)):
        model = cls(H, W)
        for f in frames:
            refined, bg_prob, _ = model.apply(f)
        assert bg_prob is None, (
            "with post-processing on, bg_prob is not copied back — returning "
            "the stale buffer instead of None hides that behind a mask of zeros")
        # fetch it explicitly for the comparison below
        outs[name] = (np.asarray(refined).copy(),
                      model.d_bg_prob.copy_to_host())

    assert np.array_equal(outs["v1"][0], outs["v2"][0]), \
        "v2's fused epilogue changed the mask; it is meant to be a pure speedup"
    assert np.allclose(outs["v1"][1], outs["v2"][1], atol=1e-6), \
        "v2's model kernel drifted from v1's"

    host = refine_mask(None, bg_prob=outs["v1"][1], do_fill=False)
    assert np.array_equal(outs["v1"][0], host), \
        "the GPU chain and the scored host chain disagree"
    assert (outs["v1"][0] == 255).any() and (outs["v1"][0] == 0).any(), \
        "degenerate mask — this test would pass on an all-zero chain"


# ── the class of bug CUDASIM structurally cannot catch ────────────────────────

@pytest.mark.skipif(os.environ.get("NUMBA_ENABLE_CUDASIM") == "1",
                    reason="the point of this test is that CUDASIM cannot fail it")
@pytest.mark.skipif(not gpu_available(), reason="needs a real CUDA device")
def test_every_kernel_actually_compiles_on_real_hardware():
    """Force nvvm to compile every kernel, and say why that is a test at all.

    `median5_tiled_kernel` declared its shared array as

        cuda.shared.array((TILE_Y + 2 * HALO, TILE_X + 2 * HALO), uint8)

    which typed the shape expression as int64 and matched no overload:

        Overload of function 'array': With argument(s):
        '(UniTuple(int64 x 2), class(uint8))': No match.

    Under CUDASIM the shape is just a Python tuple, so it passed there for as
    long as it existed while never once having run on a GPU — and v2, which
    depends on it, was in the same position. The fix was hoisting the
    arithmetic to module scope so the values are plain ints.

    A green CUDASIM suite is therefore a necessary and *not* sufficient
    condition, and this test is the marker for that. It has to be run on real
    hardware to mean anything, which is exactly the property it is asserting.
    """
    from gmm_mask.gpu import blur_kernels as bk
    from gmm_mask.gpu import post_kernels as pk

    pk.warmup()          # threshold, median naive, median tiled, dilate, erode
    bk.warmup()          # blur naive, blur tiled, composite, colour conversion

    # The compile check is `warmup()` itself: a kernel that cannot be lowered
    # raises TypingError from the launch above, which is precisely how the
    # shared-array bug surfaced. The loop below is a second, weaker guard that
    # warmup() actually launches every kernel it claims to — a kernel silently
    # dropped from warmup() would go back to being unverified.
    #
    # `overloads` is the attribute across numba versions; older ones also had
    # `definitions`, and 0.60's CUDADispatcher has only `overloads`. Probe
    # rather than assume, so this test does not itself become the thing that
    # breaks on a different numba.
    for mod, names in ((pk, ["threshold_kernel", "median5_kernel",
                             "median5_tiled_kernel", "dilate_kernel",
                             "erode_kernel"]),
                       (bk, ["blur_h_kernel", "blur_v_composite_kernel",
                             "blur_h_tiled_kernel",
                             "blur_v_composite_tiled_kernel",
                             "bgr2ycrcb_planar_kernel"])):
        for name in names:
            k = getattr(mod, name)
            compiled = getattr(k, "overloads", None)
            if compiled is None:
                compiled = getattr(k, "definitions", None)
            if compiled is None:
                continue          # this numba exposes neither; warmup() is the test
            assert len(compiled) > 0, \
                f"{name} never compiled — warmup() does not launch it"
