"""Every implementation of the model must produce the same mask.

`proposal.md` promises this twice — "three implementations: sequential Python
-> Numba CPU -> CUDA, all producing identical masks" at the 100% mark, and
agreement with `cv2.createBackgroundSubtractorMOG2()` at the 75% mark — and
until this file existed nothing in the repository checked either one. The only
surviving test compared v1 against v2, which is the narrowest pair of the six.

That gap matters more than it looks. Every speedup in this project is argued as
"same output, less time"; without a parity test the *same output* half is an
assertion. A benchmark table across four backends whose masks were never
compared is a table of four different algorithms.

Run anywhere::

    pytest tests/test_parity.py                       # CPU backends only
    NUMBA_ENABLE_CUDASIM=1 pytest tests/test_parity.py # + the CUDA backends

Sizes are deliberately tiny. `GMM_Mask_CPU` is the unvectorised specification
model — a per-pixel Python loop — so it costs roughly a second per thousand
pixels per frame, and CUDASIM runs one Python thread per CUDA thread on top of
that. 24x32 over 10 frames exercises every branch (mode creation, matching,
pruning, the sort) without making the suite something nobody runs.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2
import numpy as np
import pytest

from gmm_mask import GMM_Mask_CPU, GMM_Mask_Numba
from settings import (MOG2_BACKGROUND_RATIO, MOG2_HISTORY, MOG2_N_COMPONENTS,
                      MOG2_VAR_THRESHOLD)

H, W = 32, 48
NFRAMES = 10


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


BLOCK = 6


def traffic(n=NFRAMES, shape=(H, W), seed=0):
    """A static textured background with a bright block sweeping across it.

    Shaped like the thing being modelled rather than like noise: the block has
    to create modes, displace them, and let the old ones decay, which is what
    drives the mode sort and the replace-weakest branch. Uniform noise alone
    leaves most of the kernel unvisited, and a parity test that never runs the
    interesting path proves nothing.

    The stride is derived from `n` so the block sweeps the full width exactly
    once and never revisits a column. That detail is load-bearing: with a
    stride shorter than the block, consecutive frames overlap and only the
    leading edge reads as foreground, and once the block wraps onto a column it
    already taught the model, the mask goes empty and the comparison below
    degenerates into `all_zeros == all_zeros`.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    road = rng.integers(60, 110, (h, w, 3), dtype=np.uint8)
    stride = max(1, (w - BLOCK - 1) // max(n - 1, 1))
    out = []
    for i in range(n):
        f = road.astype(np.float32)
        x0 = min(i * stride, w - BLOCK)
        y0 = h // 3
        f[y0:y0 + BLOCK, x0:x0 + BLOCK] = 210.0
        out.append(np.ascontiguousarray(
            np.clip(f + rng.normal(0, 1.5, f.shape), 0, 255), dtype=np.uint8))
    return out


PRUNE_ALPHA = 0.2


def transient(n=20, shape=(H, W), seed=3):
    """A sequence that reaches Zivkovic's complexity-reduction branch.

    `traffic()` does not, and that matters: over its ten frames no pixel's
    active-component count ever *decreases*, so a test built on it passes
    whether or not the prune exists. That is exactly how the CuPy backend went
    on shipping with `nmodes -= 1` commented out.

    Reaching the branch needs a component that is created and then abandoned
    long enough for its weight to fall below `alpha * CT`. The default learning
    rate ramps down as 1/min(2n, history), which shrinks the *threshold* faster
    than it decays the weight, so the branch is unreachable in a short run. A
    fixed alpha fixes both. At 0.2 the object's modes are pruned around frame
    17; the test asserts the decrement rather than trusting this comment.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    road = rng.integers(70, 100, (h, w, 3), dtype=np.uint8)
    out = []
    for i in range(n):
        f = road.astype(np.float32)
        if 2 <= i < 6:                       # appears, then gone for good
            f[h // 4:3 * h // 4, w // 5:4 * w // 5] = 230.0
        out.append(np.ascontiguousarray(
            np.clip(f + rng.normal(0, 1.0, f.shape), 0, 255), dtype=np.uint8))
    return out


def run_and_watch_pruning(cls, frames, alpha=PRUNE_ALPHA, **kw):
    """Drive a backend at fixed alpha and count how often a mode was dropped."""
    model = cls(H, W, **kw)
    masks, drops, prev = [], 0, None
    for f in frames:
        mask, _, _ = model.apply(f.astype(np.float32), update_alpha=alpha)
        masks.append(np.asarray(mask).copy())
        if hasattr(model, "sync_state"):
            model.sync_state()
        cur = model.modes.copy()
        if prev is not None:
            drops += int((cur < prev).sum())
        prev = cur
    return model, masks, drops


def run(cls, frames, **kw):
    """Drive one backend over the sequence; return **every** mask.

    Every frame, not just the last one. A divergence that appears at frame 4
    and heals by frame 10 is still a divergence, and comparing only the final
    state is precisely the mistake `bench_post.py` was rewritten to stop
    making — it would be absurd to fix it there and commit it here.

    `apply` takes an HWC frame and does its own planar conversion, so the
    frames go in exactly as a decoder produced them. Passing `to_planar(...)`
    here would transpose twice and silently compare nonsense that happens to
    have a plausible shape.
    """
    model = cls(H, W, **kw)
    masks, probs = [], []
    for f in frames:
        mask, bg_prob, _ = model.apply(f.astype(np.float32))
        masks.append(np.asarray(mask).copy())
        probs.append(None if bg_prob is None else np.asarray(bg_prob).copy())
    if hasattr(model, "sync_state"):
        model.sync_state()
    return model, masks, probs


def assert_all_frames_equal(a, b, label):
    diffs = [(i, int((x != y).sum())) for i, (x, y) in enumerate(zip(a, b))
             if not np.array_equal(x, y)]
    assert not diffs, (
        f"{label}: diverges on {len(diffs)} of {len(a)} frames; "
        f"first at frame {diffs[0][0]} ({diffs[0][1]} px)")


def assert_not_degenerate(mask):
    assert (mask == 255).any() and (mask == 0).any(), (
        "mask is uniform — this comparison would pass on two broken backends")


# ── the specification against the fast CPU backend ────────────────────────────

def test_numba_matches_the_sequential_specification():
    """`GMM_Mask_CPU` is the readable transliteration of Zivkovic; Numba is the
    one that is actually fast. If they disagree, the readable one is the truth
    and every number in the report was measured on the other one."""
    frames = traffic()
    _, m_cpu, p_cpu = run(GMM_Mask_CPU, frames)
    _, m_nb, p_nb = run(GMM_Mask_Numba, frames)

    assert_not_degenerate(m_cpu[-1])
    assert_all_frames_equal(m_cpu, m_nb, "sequential vs Numba")
    for i, (a, b) in enumerate(zip(p_cpu, p_nb)):
        assert np.allclose(a, b, atol=1e-6), f"bg_prob differs at frame {i}"


def test_the_model_state_agrees_and_not_just_the_mask():
    """Two backends can agree on the thresholded mask while their mixtures have
    drifted apart — the mask is a comparison against Tb, and a small difference
    in variance often lands on the same side of it. The state is where drift
    shows up first, so it is the earlier warning."""
    frames = traffic()
    a, _, _ = run(GMM_Mask_CPU, frames)
    b, _, _ = run(GMM_Mask_Numba, frames)
    for name in ("weights", "means", "vars"):
        assert np.allclose(getattr(a, name), getattr(b, name), atol=1e-5), \
            f"{name} drifted between the sequential and Numba backends"
    assert np.array_equal(a.modes, b.modes)


# ── the CUDA backends against the CPU reference ───────────────────────────────

@requires_gpu
@pytest.mark.parametrize("name", ["cuda", "cuda_v1", "cuda_v2"])
def test_cuda_backends_match_the_cpu_reference_in_model_state_too(name):
    """Masks agreeing is the weaker half of the claim.

    Two backends can agree on every thresholded mask while their mixtures have
    drifted apart, because the mask is a comparison against Tb and a small
    difference in variance usually lands on the same side of it. The state is
    where drift appears first, so it is the earlier warning — and until this
    test existed, `proposal.md` claimed agreement "in mask and model state"
    while only CPU-vs-Numba compared state at all.
    """
    from gmm_mask import GMM_Mask_CUDA, GMM_Mask_CUDA_v1, GMM_Mask_CUDA_v2
    cls, kw = {
        "cuda": (GMM_Mask_CUDA, {}),
        "cuda_v1": (GMM_Mask_CUDA_v1, {"post": False}),
        "cuda_v2": (GMM_Mask_CUDA_v2, {"post": False}),
    }[name]
    if cls is None:
        pytest.skip("CUDA backends unavailable")

    frames = traffic()
    ref, _, _ = run(GMM_Mask_Numba, frames)
    gpu, _, _ = run(cls, frames, **kw)      # run() calls sync_state() for us

    for field in ("weights", "means", "vars"):
        a, b = getattr(ref, field), getattr(gpu, field)
        assert np.allclose(a, b, atol=1e-5), (
            f"{name}: {field} drifted from the Numba reference "
            f"(max |delta| {np.abs(a - b).max():.3g})")
    assert np.array_equal(ref.modes, gpu.modes), (
        f"{name}: the per-pixel active-component count diverged — that is the "
        "complexity-reduction rule differing, not rounding")


@requires_gpu
@pytest.mark.parametrize("name", ["cuda", "cuda_v1", "cuda_v2"])
def test_cuda_backends_match_the_cpu_reference(name):
    """With post-processing off, every CUDA backend is meant to be the same
    model, not a similar one. v1 and v2 are compared here with `post=False`
    precisely so this is a test of the *model* kernel; their post-processing
    chain is already covered in `test_post_chain.py`."""
    from gmm_mask import GMM_Mask_CUDA, GMM_Mask_CUDA_v1, GMM_Mask_CUDA_v2
    cls, kw = {
        "cuda": (GMM_Mask_CUDA, {}),
        "cuda_v1": (GMM_Mask_CUDA_v1, {"post": False}),
        "cuda_v2": (GMM_Mask_CUDA_v2, {"post": False}),
    }[name]
    if cls is None:
        pytest.skip("CUDA backends unavailable")

    frames = traffic()
    _, m_ref, p_ref = run(GMM_Mask_Numba, frames)
    _, m_gpu, p_gpu = run(cls, frames, **kw)

    assert_not_degenerate(m_ref[-1])
    assert_all_frames_equal(m_ref, m_gpu, f"Numba vs {name}")
    for i, (a, b) in enumerate(zip(p_ref, p_gpu)):
        assert np.allclose(a, b, atol=1e-5), f"{name}: bg_prob differs at frame {i}"


# ── against OpenCV, the thing being reimplemented ─────────────────────────────

def test_agreement_with_opencvs_own_mog2():
    """The 75% deliverable, and the strongest correctness claim in the project.

    Measured, not assumed. On this synthetic sequence our sequential model is
    **bit-identical** to `cv2.createBackgroundSubtractorMOG2` — 0 of 30 720
    pixels differ over 20 frames — and it stays exact under stress: 0 of 92 160
    on heavy noise at 60 frames, 0 of 204 800 at 64x80. So the assertion here
    is equality, not a tolerance. A tolerance would let a real regression hide
    inside the slack.

    On real footage it is *almost* exact rather than exact: 22 pixels of
    1 536 000 on `LTSSUD-Test.mp4` (0.0014%), and all 22 fall in a single
    frame; 1 pixel of 1 536 000 on `TestStableBackground.mp4`. That is the
    float32 boundary showing itself — OpenCV accumulates in a different order
    and contracts its own FMAs, so a pixel sitting within an ulp of
    `Tb * var` can land on either side of the comparison. Synthetic frames put
    almost nothing that close to the threshold; camera noise does.

    Report both numbers. "Bit-exact on synthetic input, 0.0014% disagreement on
    video" is a true and unusually strong statement; "bit-exact" unqualified is
    not, and the difference is one frame of one clip.
    """
    frames = traffic(n=20)
    ours = GMM_Mask_CPU(H, W)
    cv = cv2.createBackgroundSubtractorMOG2(
        history=int(MOG2_HISTORY),
        varThreshold=float(MOG2_VAR_THRESHOLD),
        detectShadows=False)
    cv.setNMixtures(int(MOG2_N_COMPONENTS))
    cv.setBackgroundRatio(float(MOG2_BACKGROUND_RATIO))

    disagree = total = 0
    for f in frames:
        m_ours, _, _ = ours.apply(f.astype(np.float32))
        m_cv = cv.apply(f)
        disagree += int((np.asarray(m_ours) != m_cv).sum())
        total += m_cv.size

    assert_not_degenerate(np.asarray(m_ours))
    assert disagree == 0, (
        f"{disagree} of {total} pixels ({disagree / total:.4%}) disagree with "
        "cv2.BackgroundSubtractorMOG2. This sequence has always been exact; a "
        "nonzero count here is a change in the update rule, not rounding.")


# ── the bug this file was written after ───────────────────────────────────────

@pytest.mark.parametrize("name", ["cpu", "numba"])
def test_background_image_is_callable(name):
    """Three GPU subclasses overrode `background_image()` and called
    `super().background_image()`, which no base class defined — every one of
    them raised AttributeError on first call. Nothing in the pipeline calls it,
    so nothing caught it. This is the test that would have."""
    cls = {"cpu": GMM_Mask_CPU, "numba": GMM_Mask_Numba}[name]
    model, _, _ = run(cls, traffic(n=3))
    bg = model.background_image()
    assert bg.shape == (H, W, 3) and bg.dtype == np.uint8
    assert bg.any(), "the learned background came back entirely black"

    # The picture must be the background, not the moving object. The block is
    # at 210 on a road of 60-110; if the reconstruction picked the wrong modes
    # the road would come back bright.
    assert bg.mean() < 150, "background_image looks like the foreground"

    # Round, not truncate. Halving a background of ~85 lands mid-integer often
    # enough that truncation shows up as a systematic downward bias.
    ref = np.rint(model.means[0]).astype(np.uint8).transpose(1, 2, 0)
    assert abs(float(bg.mean()) - float(ref.mean())) < 12


@requires_gpu
@pytest.mark.parametrize("name", ["cuda", "cuda_v1"])
def test_gpu_background_image_syncs_device_state_first(name):
    from gmm_mask import GMM_Mask_CUDA, GMM_Mask_CUDA_v1
    cls = {"cuda": GMM_Mask_CUDA, "cuda_v1": GMM_Mask_CUDA_v1}[name]
    if cls is None:
        pytest.skip("CUDA backends unavailable")
    model, _, _ = run(cls, traffic(n=3))
    bg = model.background_image()
    assert bg.shape == (H, W, 3) and bg.any()
    assert bg.mean() < 150


def cupy_available():
    try:
        from gmm_mask import GMM_Mask_CuPy
        if GMM_Mask_CuPy is None:
            return False
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not cupy_available(), reason="cupy or a CUDA device is unavailable")
def test_cupy_matches_the_cpu_reference_including_the_prune():
    """The backend that was silently a different algorithm.

    `step_kernel_cp_v1.cu` had Zivkovic's complexity-reduction step commented
    out — the component prune and the `nmodes` decrement both. So CuPy kept
    components the other three backends delete, and nothing noticed, because
    no test compared it at all.

    Restoring it is not enough; the test has to *reach* the branch. The first
    version of this test used `traffic()`, where no pixel's mode count ever
    decreases, so it would have passed against the broken kernel too. The
    assertion below on `drops` is the guard against that regression in the
    test itself.
    """
    from gmm_mask import GMM_Mask_CuPy

    frames = transient()
    ref, m_ref, drops = run_and_watch_pruning(GMM_Mask_Numba, frames)
    assert drops > 0, (
        "the fixture never pruned a component — this test cannot distinguish "
        "a correct kernel from one with the prune commented out")

    cp_model, m_cp, cp_drops = run_and_watch_pruning(GMM_Mask_CuPy, frames)

    assert_not_degenerate(m_ref[3])
    assert_all_frames_equal(m_ref, m_cp, "Numba vs CuPy")
    assert cp_drops == drops, (
        f"CuPy pruned {cp_drops} components where the reference pruned "
        f"{drops} — the complexity-reduction rule differs")
    for field in ("weights", "means", "vars"):
        a, b = getattr(ref, field), getattr(cp_model, field)
        assert np.allclose(a, b, atol=1e-5), (
            f"CuPy: {field} drifted (max |delta| {np.abs(a - b).max():.3g})")
    assert np.array_equal(ref.modes, cp_model.modes)


@requires_gpu
@pytest.mark.parametrize("name", ["cuda", "cuda_v1", "cuda_v2"])
def test_cuda_backends_agree_on_the_prune_branch_too(name):
    """`traffic()` never prunes, so the tests above never checked this path.

    The complexity-reduction rule is the one place the backends were known to
    have diverged once, and it is per-pixel branchy — exactly the kind of thing
    a transliteration gets wrong.
    """
    from gmm_mask import GMM_Mask_CUDA, GMM_Mask_CUDA_v1, GMM_Mask_CUDA_v2
    cls, kw = {
        "cuda": (GMM_Mask_CUDA, {}),
        "cuda_v1": (GMM_Mask_CUDA_v1, {"post": False}),
        "cuda_v2": (GMM_Mask_CUDA_v2, {"post": False}),
    }[name]
    if cls is None:
        pytest.skip("CUDA backends unavailable")

    frames = transient()
    ref, m_ref, drops = run_and_watch_pruning(GMM_Mask_Numba, frames)
    assert drops > 0, "fixture does not reach the prune branch"

    gpu, m_gpu, gpu_drops = run_and_watch_pruning(cls, frames, **kw)
    assert_all_frames_equal(m_ref, m_gpu, f"Numba vs {name} (pruning sequence)")
    assert gpu_drops == drops, (
        f"{name} pruned {gpu_drops} components against the reference's {drops}")
    assert np.array_equal(ref.modes, gpu.modes)
