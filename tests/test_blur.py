"""Kernel 2 must equal `cv2.GaussianBlur` + `cv2.copyTo`, exactly.

The host chain in `utils.post_processing` is the specification. Every assertion
below is `np.array_equal` — there is no tolerance anywhere, because there is no
floating-point arithmetic anywhere in the blur path. If one of these fails,
something is genuinely wrong; there is no slack for a bug to hide in.

    pytest tests/test_blur.py                        # the host claims
    NUMBA_ENABLE_CUDASIM=1 pytest tests/test_blur.py # + the kernels
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2
import numpy as np
import pytest

from gmm_mask.gpu.blur_kernels import (BLUR_KSIZE, BLUR_R, BLUR_SIGMA,
                                       BLUR_TILE_X, BLUR_TILE_Y,
                                       blur_grid_for, blur_reference,
                                       gaussian_kernel_q8)
from utils.post_processing import background_blur

H, W = 32, 48
KQ = gaussian_kernel_q8()
BLOCK = (BLUR_TILE_X, BLUR_TILE_Y)


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


def frames(shape=(H, W)):
    """The cases that have historically broken a tiled stencil."""
    rng = np.random.default_rng(7)
    h, w = shape
    out = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(4)]
    out.append(np.zeros((h, w, 3), np.uint8))
    out.append(np.full((h, w, 3), 255, np.uint8))
    lit = np.zeros((h, w, 3), np.uint8)
    lit[0] = lit[-1] = 255
    lit[:, 0] = lit[:, -1] = 255
    out.append(lit)                       # every border condition at once
    return out


def blobby(seed=0, shape=(H, W)):
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (9, 9), 0)
    return (field > 0.52).astype(np.uint8) * 255


# ── the host claims: no GPU, no CUDASIM, and they carry the whole argument ────

def test_the_q8_kernel_is_opencvs_own_fixed_point_kernel():
    """Recover OpenCV's actual integer kernel and compare against ours.

    A single lit column, constant down the image, makes the vertical pass the
    identity, so the output reads back `round(KQ[j]*255/256) == KQ[j]`. The
    probe is only exact while `max(KQ) < 128`; assert that rather than trust
    it, because at sigma around 0.8 the centre tap reaches 128 and the probe
    starts quietly lying.
    """
    assert KQ.sum() == 256
    assert KQ.max() < 128, "impulse probe is not valid for this kernel"

    n = 4 * BLUR_KSIZE
    img = np.zeros((n, n), np.uint8)
    img[:, n // 2] = 255
    row = cv2.GaussianBlur(img, (BLUR_KSIZE, BLUR_KSIZE), BLUR_SIGMA)[n // 2]
    probe = row[n // 2 - BLUR_R: n // 2 + BLUR_R + 1].astype(np.int32)
    assert np.array_equal(probe, KQ), f"OpenCV uses {probe}, we derive {KQ}"


def test_per_tap_rounding_is_not_what_opencv_does():
    """The obvious wrong implementation, pinned as wrong.

    Rounding each tap independently is what anyone writing this from scratch
    reaches for, it also sums to 256 at (15, 5.0), and it disagrees with
    OpenCV. Without this test someone simplifies the cumulative derivation back
    into `np.rint(k*256)` and every blur number in the report shifts by one.
    """
    k = cv2.getGaussianKernel(BLUR_KSIZE, BLUR_SIGMA).ravel()
    per_tap = np.rint(k * 256).astype(np.int32)
    assert not np.array_equal(per_tap, KQ)
    assert (np.abs(per_tap - KQ) > 0).sum() >= 2


@pytest.mark.parametrize("shape", [(H, W), (37, 53), (15, 15), (8, 9)])
def test_the_reference_is_bit_exact_with_cv2_gaussianblur(shape):
    """The tripwire for an OpenCV version or build change.

    This is the one test that must be re-run on the Colab image before any
    number is quoted from it: everything downstream is checked against
    `blur_reference`, and `blur_reference` is only meaningful while this holds.
    """
    for img in frames(shape):
        ours = blur_reference(img, KQ)
        theirs = cv2.GaussianBlur(img, (BLUR_KSIZE, BLUR_KSIZE), BLUR_SIGMA)
        assert np.array_equal(ours, theirs), (
            f"{int((ours != theirs).sum())} pixels differ at {shape}")


def test_an_ideal_float_gaussian_would_differ_from_opencv_by_one():
    """Documents whose error the +/-1 is.

    A float64 convolution with correct rounding is *not* what cv2 computes; it
    differs on a sixth of the frame by one grey level. We reproduce OpenCV, so
    we inherit that departure — and saying so is the honest version of "bit
    exact with cv2". The lower bound also fails if someone quietly makes
    `blur_reference` floating point, which would otherwise look like a cleanup.
    """
    img = frames()[0]
    k = cv2.getGaussianKernel(BLUR_KSIZE, BLUR_SIGMA).ravel()
    ideal = cv2.sepFilter2D(img.astype(np.float64), cv2.CV_64F, k, k,
                            borderType=cv2.BORDER_REFLECT_101)
    ideal = np.floor(ideal + 0.5).astype(np.uint8)
    ours = blur_reference(img, KQ)
    diff = np.abs(ideal.astype(np.int32) - ours.astype(np.int32))
    assert diff.max() <= 1
    assert (diff != 0).mean() > 0.05, (
        "the reference now agrees with an ideal Gaussian, which means it is no "
        "longer reproducing OpenCV's fixed-point path")


def test_gaussian_kernel_q8_rejects_inputs_opencv_treats_specially():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            gaussian_kernel_q8(15, bad)
    for bad in (14, 1):
        with pytest.raises(ValueError):
            gaussian_kernel_q8(bad, 5.0)


# ── the kernels ───────────────────────────────────────────────────────────────

@requires_gpu
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 15, 64])
def test_reflect101_matches_cv2_for_every_width_including_one(n):
    """`n == 1` is the case that hangs.

    Without the early return, -7 reflects to 7, 7 reflects to 2*(1-1)-7 = -7,
    and the kernel spins forever on a one-pixel-wide image. Nothing in the
    pipeline produces one, which is exactly why it would ship.
    """
    from gmm_mask.gpu.blur_kernels import _reflect101
    fn = getattr(_reflect101, "py_func", _reflect101)

    row = np.arange(n, dtype=np.uint8).reshape(1, n)
    padded = cv2.copyMakeBorder(row, 0, 0, BLUR_R, BLUR_R,
                                cv2.BORDER_REFLECT_101)[0]
    ours = [row[0, fn(i - BLUR_R, n)] for i in range(n + 2 * BLUR_R)]
    assert np.array_equal(np.array(ours, np.uint8), padded)


def _run_blur(img, mask, tiled):
    from numba import cuda
    import gmm_mask.gpu.blur_kernels as bk

    h, w = img.shape[:2]
    grid = blur_grid_for(h, w)
    d_kq = cuda.to_device(KQ)
    d_src = cuda.to_device(np.ascontiguousarray(img))
    d_msk = cuda.to_device(np.ascontiguousarray(mask))
    d_hor = cuda.device_array((h, w, 3), np.uint16)
    d_out = cuda.device_array((h, w, 3), np.uint8)

    kh = bk.blur_h_tiled_kernel if tiled else bk.blur_h_kernel
    kv = (bk.blur_v_composite_tiled_kernel if tiled
          else bk.blur_v_composite_kernel)
    kh[grid, BLOCK](d_src, d_hor, d_kq)
    kv[grid, BLOCK](d_hor, d_src, d_msk, d_out, d_kq)
    return d_out.copy_to_host()


@requires_gpu
@pytest.mark.parametrize("tiled", [False, True], ids=["naive", "tiled"])
def test_blur_matches_cv2_when_nothing_is_masked(tiled):
    """An all-zero mask means blur everywhere, so this isolates the blur."""
    img = frames()[0]
    out = _run_blur(img, np.zeros((H, W), np.uint8), tiled)
    ref = cv2.GaussianBlur(img, (BLUR_KSIZE, BLUR_KSIZE), BLUR_SIGMA)
    assert np.array_equal(out, ref), f"{int((out != ref).sum())} pixels differ"


@requires_gpu
@pytest.mark.parametrize("tiled", [False, True], ids=["naive", "tiled"])
def test_composite_matches_the_host_background_blur(tiled):
    img = frames()[0]
    mask = blobby()
    assert (mask == 0).any() and (mask == 255).any()

    out = _run_blur(img, mask, tiled)
    ref = background_blur(img, mask, BLUR_KSIZE, BLUR_SIGMA)
    assert np.array_equal(out, ref), f"{int((out != ref).sum())} pixels differ"
    assert (out != img).any(), "nothing was blurred — degenerate comparison"


@requires_gpu
@pytest.mark.parametrize("tiled", [False, True], ids=["naive", "tiled"])
def test_every_non_zero_mask_value_keeps_the_pixel(tiled):
    """`cv2.copyTo` selects on non-zero, not on 255.

    `fill_holes` emits only 0 and 255 today, so this never fires in production
    — and that is the point: the kernel must not silently acquire a dependency
    on it that a later change to the mask chain would break.
    """
    rng = np.random.default_rng(3)
    img = frames()[0]
    mask = rng.choice(np.array([0, 1, 127, 255], np.uint8), (H, W))
    out = _run_blur(img, np.ascontiguousarray(mask), tiled)
    ref = background_blur(img, mask, BLUR_KSIZE, BLUR_SIGMA)
    assert np.array_equal(out, ref)


@requires_gpu
@pytest.mark.parametrize("tiled", [False, True], ids=["naive", "tiled"])
def test_the_border_is_reflect_101_and_not_replicate(tiled):
    """A lit first column: REPLICATE and REFLECT_101 differ here and nowhere
    else, so a wrong border shows up as a seven-pixel frame around the image
    and is invisible in every other test."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:, 0] = 255
    out = _run_blur(img, np.zeros((H, W), np.uint8), tiled)
    ref = cv2.GaussianBlur(img, (BLUR_KSIZE, BLUR_KSIZE), BLUR_SIGMA)
    assert np.array_equal(out[:, :BLUR_R + 1], ref[:, :BLUR_R + 1])
    assert out[:, :BLUR_R + 1].any(), "degenerate: the border region is empty"


@requires_gpu
def test_tiled_equals_naive_on_a_size_that_is_not_a_tile_multiple():
    """37x53 against a 32x16 block: ragged edge blocks in both axes, which is
    where a halo bug lives if it lives anywhere."""
    img = frames((37, 53))[0]
    mask = blobby(1, (37, 53))
    a = _run_blur(img, mask, tiled=False)
    b = _run_blur(img, mask, tiled=True)
    assert np.array_equal(a, b), f"{int((a != b).sum())} pixels differ"
    assert np.array_equal(a, background_blur(img, mask, BLUR_KSIZE, BLUR_SIGMA))


@requires_gpu
def test_the_colour_conversion_kernel_matches_cv2_exactly():
    from numba import cuda
    from gmm_mask.gpu.blur_kernels import bgr2ycrcb_planar_kernel

    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    img[0] = 0
    img[1] = 255                          # both saturation corners
    d_out = cuda.device_array((3, H, W), np.float32)
    bgr2ycrcb_planar_kernel[blur_grid_for(H, W), BLOCK](
        cuda.to_device(img), d_out, True)

    ref = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).transpose(2, 0, 1)
    assert np.array_equal(d_out.copy_to_host(), ref.astype(np.float32)), (
        "a single grey level of drift here changes what the model sees, and "
        "every quality number in the report with it")


@requires_gpu
def test_the_colour_conversion_kernel_passes_bgr_through():
    from numba import cuda
    from gmm_mask.gpu.blur_kernels import bgr2ycrcb_planar_kernel

    rng = np.random.default_rng(6)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    d_out = cuda.device_array((3, H, W), np.float32)
    bgr2ycrcb_planar_kernel[blur_grid_for(H, W), BLOCK](
        cuda.to_device(img), d_out, False)
    assert np.array_equal(d_out.copy_to_host(),
                          img.transpose(2, 0, 1).astype(np.float32))


# ── end to end: the device ingest must not change the mask ───────────────────

def _traffic_bgr(n=8, shape=(H, W), seed=11):
    rng = np.random.default_rng(seed)
    h, w = shape
    road = rng.integers(60, 110, (h, w, 3), dtype=np.uint8)
    out = []
    stride = max(1, (w - 6 - 1) // max(n - 1, 1))
    for i in range(n):
        f = road.astype(np.float32)
        x0 = min(i * stride, w - 6)
        f[h // 3:h // 3 + 6, x0:x0 + 6] = 210.0
        out.append(np.ascontiguousarray(
            np.clip(f + rng.normal(0, 1.5, f.shape), 0, 255), dtype=np.uint8))
    return out


@requires_gpu
@pytest.mark.parametrize("name", ["v1", "v2"])
def test_the_device_ingest_produces_the_same_mask_as_the_host_path(name):
    """The test that protects every quality number in the report.

    Old path: host `cvtColor`, then `apply()` — which does its own planar
    conversion, so it is handed the HWC frame. Passing `to_planar(...)` here
    would transpose twice and compare nonsense that happens to have the right
    shape.

    New path: `mask_from_bgr()` uploads BGR and converts on the device. If the
    two masks differ by one pixel, the F1 quoted for this pipeline was measured
    on a different pipeline.
    """
    from gmm_mask import GMM_Mask_CUDA_v1, GMM_Mask_CUDA_v2
    cls = {"v1": GMM_Mask_CUDA_v1, "v2": GMM_Mask_CUDA_v2}[name]
    if cls is None:
        pytest.skip("CUDA backends unavailable")

    frames_bgr = _traffic_bgr()

    old = cls(H, W)
    for f in frames_bgr:
        ycrcb = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        m_old, _, _ = old.apply(ycrcb.astype(np.float32))
    m_old = np.asarray(m_old).copy()

    new = cls(H, W)
    for f in frames_bgr:
        m_new = new.mask_from_bgr(f)
    m_new = np.asarray(m_new).copy()

    assert (m_new == 255).any() and (m_new == 0).any()
    assert np.array_equal(m_old, m_new), (
        f"{name}: {int((m_old != m_new).sum())} pixels differ between the host "
        "ingest and the device ingest")


@requires_gpu
def test_v1_and_v2_composites_match_the_host_pipeline():
    """The whole chain, against the chain `eval_highway.py` actually scored."""
    from gmm_mask import GMM_Mask_CUDA_v1, GMM_Mask_CUDA_v2
    from utils.post_processing import fill_holes
    if GMM_Mask_CUDA_v1 is None:
        pytest.skip("CUDA backends unavailable")

    frames_bgr = _traffic_bgr()
    outs = {}
    for name, cls in (("v1", GMM_Mask_CUDA_v1), ("v2", GMM_Mask_CUDA_v2)):
        model = cls(H, W)
        for f in frames_bgr:
            refined = model.mask_from_bgr(f)
            filled = fill_holes(refined)
            out = model.composite(filled)
        outs[name] = (filled.copy(), out.copy())

    assert np.array_equal(outs["v1"][0], outs["v2"][0]), \
        "v2's tiled median changed the mask; it is meant to be a pure speedup"
    assert np.array_equal(outs["v1"][1], outs["v2"][1]), \
        "v2's tiled blur changed the composite"

    host = background_blur(frames_bgr[-1], outs["v1"][0], BLUR_KSIZE, BLUR_SIGMA)
    assert np.array_equal(outs["v1"][1], host), (
        f"{int((outs['v1'][1] != host).sum())} pixels differ from the host "
        "composite")


@requires_gpu
def test_the_no_fill_path_composites_against_the_device_mask():
    """`--no-fill` keeps the mask on the device. The saving is real only if the
    composite then uses that mask rather than a stale host buffer."""
    from gmm_mask import GMM_Mask_CUDA_v2
    if GMM_Mask_CUDA_v2 is None:
        pytest.skip("CUDA backends unavailable")

    frames_bgr = _traffic_bgr()

    resident = GMM_Mask_CUDA_v2(H, W)
    for f in frames_bgr:
        resident.mask_from_bgr(f, to_host=False)
    out_resident = resident.composite().copy()

    viahost = GMM_Mask_CUDA_v2(H, W)
    for f in frames_bgr:
        mask = viahost.mask_from_bgr(f)
    out_viahost = viahost.composite(mask).copy()

    assert np.array_equal(out_resident, out_viahost)
    assert np.array_equal(
        out_resident, background_blur(frames_bgr[-1], mask, BLUR_KSIZE, BLUR_SIGMA))


@requires_gpu
def test_composite_before_mask_from_bgr_is_an_error_not_a_wrong_picture():
    from gmm_mask import GMM_Mask_CUDA_v2
    if GMM_Mask_CUDA_v2 is None:
        pytest.skip("CUDA backends unavailable")
    with pytest.raises(RuntimeError):
        GMM_Mask_CUDA_v2(H, W).composite()


@requires_gpu
@pytest.mark.parametrize("name", ["v1", "v2"])
def test_the_blur_uses_its_own_launch_geometry(name):
    """Borrowing the model's 32x8 block would leave half of every shared tile
    unloaded, and the result would still look like a blur. Pin the production
    attributes, not just a launch written inside a test."""
    from gmm_mask import GMM_Mask_CUDA_v1, GMM_Mask_CUDA_v2
    cls = {"v1": GMM_Mask_CUDA_v1, "v2": GMM_Mask_CUDA_v2}[name]
    if cls is None:
        pytest.skip("CUDA backends unavailable")
    m = cls(H, W)
    assert m.blur_block == (BLUR_TILE_X, BLUR_TILE_Y)
    assert m.blur_block != m.block
    assert m.blur_grid == blur_grid_for(H, W)
