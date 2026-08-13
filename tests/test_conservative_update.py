"""The stationary-subject case: a background model must not eat what stops moving.

Plain MOG2 updates every pixel unconditionally, so a subject who holds still is
folded into the background within a couple of seconds and stops being detected.
That is the failure the webcam demo actually hits, and no post-processing can
undo it — the mask it would be refining no longer contains the person.

These tests are synthetic on purpose. Compositing a subject onto a background
is the only way to get a pixel-exact ground truth for "is the subject still
detected on frame 200", and that is the whole question. `docs/conservative.md`
carries the same experiment against real imagery.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pytest

from gmm import GMM_CPU, GMM_CPU_NUMBA
from gmm.mog2_common import to_planar, opencv_reference
from settings import MOG2_N_COMPONENTS

H, W = 60, 80
BOX = (slice(20, 45), slice(25, 55))       # the "subject"
N_WARMUP = 30                              # clean-plate frames, subject absent
N_TOTAL = 160


def sequence(present_from_zero, seed=0):
    """Textured static background, subject pasted in, sensor noise on both.

    The noise matters: without it a motionless paste is bit-identical every
    frame, which would make the subject invisible to *any* background model and
    turn this into a test of nothing.
    """
    rng = np.random.default_rng(seed)
    bg = rng.integers(40, 90, (H, W, 3), dtype=np.uint8)
    subject = rng.integers(150, 210, (H, W, 3), dtype=np.uint8)
    for i in range(N_TOTAL):
        frame = bg.astype(np.float32)
        gt = np.zeros((H, W), bool)
        if present_from_zero or i >= N_WARMUP:
            frame[BOX] = subject[BOX]
            gt[BOX] = True
        frame = np.clip(frame + rng.normal(0, 2.5, frame.shape), 0, 255).astype(np.uint8)
        yield frame, gt


def subject_iou(model_cls, present_from_zero, conservative, at=N_TOTAL - 1):
    """IoU on the subject box at frame `at` — i.e. long after it stopped moving."""
    model = None
    iou = 0.0
    for i, (frame, gt) in enumerate(sequence(present_from_zero)):
        if model is None:
            model = model_cls(frame, n_components=MOG2_N_COMPONENTS,
                              conservative=conservative)
        mask, _ = model.step(to_planar(frame))
        if i == at:
            p = np.asarray(mask) == 255
            iou = (p & gt).sum() / max((p | gt).sum(), 1)
    return iou


def test_plain_mog2_absorbs_a_subject_that_stops_moving():
    """The bug, pinned. If this ever starts passing, MOG2 stopped being MOG2."""
    assert subject_iou(GMM_CPU_NUMBA, False, conservative=False) < 0.1


def test_conservative_update_holds_a_stationary_subject():
    """The fix. 130 frames after the subject last moved it is still there."""
    assert subject_iou(GMM_CPU_NUMBA, False, conservative=True) > 0.9


def test_conservative_update_still_needs_a_clean_plate():
    """A subject present in frame 0 is inside the model before anything runs.

    Conservative update protects what was *detected*; it cannot protect what was
    never detected in the first place. This is the LTSSUD-Test.mp4 case, and the
    reason `main.py` grew a --clean-plate option rather than just a flag.
    """
    assert subject_iou(GMM_CPU_NUMBA, True, conservative=True) < 0.1


def test_protection_releases_when_the_subject_leaves():
    """The subject walks away and the pixel comes back — no ghost here.

    Once the subject moves away the pixel matches the background it was frozen
    at and starts updating again on the next frame. This is the case the rule
    handles cleanly; it is *not* a general "no everlasting ghost" guarantee.
    A background that genuinely changed while protected, and a global
    appearance change, are separate problems — see
    `test_a_global_exposure_change_does_not_latch_the_frame` and
    `MOG2_PROTECT_EXIT`.
    """
    model = None
    for i, (frame, _) in enumerate(sequence(False)):
        if model is None:
            model = GMM_CPU_NUMBA(frame, n_components=MOG2_N_COMPONENTS,
                                  conservative=True)
        mask, _ = model.step(to_planar(frame))
    # Without this the test passes when the feature is entirely deleted: if the
    # subject was never protected there is nothing to release, and an empty
    # mask satisfies the assertion below trivially.
    assert (np.asarray(mask)[BOX] == 255).mean() > 0.9, "subject was never protected"

    # subject gone from here on — same background, same noise stream
    rng = np.random.default_rng(99)
    bg = np.random.default_rng(0).integers(40, 90, (H, W, 3), dtype=np.uint8)
    for _ in range(40):
        frame = np.clip(bg + rng.normal(0, 2.5, bg.shape), 0, 255).astype(np.uint8)
        mask, _ = model.step(to_planar(frame))
    assert (np.asarray(mask)[BOX] == 255).mean() < 0.1


@pytest.mark.parametrize("conservative", [False, True])
def test_python_spec_and_numba_agree(conservative):
    """The Numba kernel is a transliteration of GMM_cpu; conservative or not.

    The whole sequence, not a `[:12]` slice: the subject does not enter until
    frame 30, so a short prefix compares two models looking at empty background
    and never exercises the branch the parameter is there to test.
    """
    seq = list(sequence(False))
    a = GMM_CPU(seq[0][0], n_components=MOG2_N_COMPONENTS, conservative=conservative)
    b = GMM_CPU_NUMBA(seq[0][0], n_components=MOG2_N_COMPONENTS, conservative=conservative)
    protected = 0
    for frame, _ in seq:
        ma, _ = a.step(to_planar(frame))
        mb, _ = b.step(to_planar(frame))
        assert np.array_equal(np.asarray(ma), np.asarray(mb))
        protected += int((np.asarray(ma)[BOX] == 255).mean() > 0.9)
    if conservative:
        assert protected > N_TOTAL // 2, "the branch under test never ran"


def test_conservative_off_is_still_bit_exact_with_opencv():
    """The parity claim must survive the feature that breaks it when enabled."""
    ref = opencv_reference()
    model = None
    for frame, _ in list(sequence(False))[:20]:
        if model is None:
            model = GMM_CPU_NUMBA(frame, n_components=MOG2_N_COMPONENTS)
        expected = ref.apply(frame)
        mask, _ = model.step(to_planar(frame))
        assert np.array_equal(np.asarray(mask), expected)


def test_frozen_pixels_really_are_frozen():
    """alpha = 0 is not enough on its own — prune has to go to zero with it.

    prune is -alpha*CT, precomputed on the host. Leaving it at its unprotected
    value would keep decaying the weights of a protected pixel every frame and
    eventually prune its modes away, which is the slow version of the same bug.
    """
    seq = list(sequence(False))
    model = GMM_CPU_NUMBA(seq[0][0], n_components=MOG2_N_COMPONENTS, conservative=True)
    for frame, _ in seq[:N_WARMUP + 20]:
        model.step(to_planar(frame))
    before = (model.weights[:, BOX[0], BOX[1]].copy(),
              model.means[:, :, BOX[0], BOX[1]].copy(),
              model.vars[:, BOX[0], BOX[1]].copy())
    for frame, _ in seq[N_WARMUP + 20:]:
        model.step(to_planar(frame))
    protected = np.asarray(model.mask)[BOX] == 255
    assert protected.mean() > 0.9, "subject should still be protected"
    for now, then in zip((model.weights[:, BOX[0], BOX[1]],
                          model.means[:, :, BOX[0], BOX[1]],
                          model.vars[:, BOX[0], BOX[1]]), before):
        assert np.array_equal(now[..., protected], then[..., protected])


def test_gpu_kernels_implement_conservative_too():
    """Deleting the conservative branch from either GPU kernel left the whole
    suite green: nothing drove them with `conservative=True`.

    Runs under NUMBA_ENABLE_CUDASIM=1 as well, which is how it gets exercised
    on a machine with no device.
    """
    import pytest
    from gmm import GMM_CUDA, GMM_CUPY
    from tests.test_smoke_models import cupy_is_mocked

    backends = [c for c in (GMM_CUDA, GMM_CUPY) if c is not None]
    if GMM_CUPY in backends and cupy_is_mocked():
        backends.remove(GMM_CUPY)
    if not backends:
        pytest.skip("no GPU backend available")

    seq = list(sequence(False))
    spec = GMM_CPU_NUMBA(seq[0][0], n_components=MOG2_N_COMPONENTS, conservative=True)
    for frame, _ in seq:
        spec_mask, _ = spec.step(to_planar(frame))
    spec_mask = np.asarray(spec_mask).copy()
    assert (spec_mask[BOX] == 255).mean() > 0.9, "the reference is not protecting"

    for cls in backends:
        m = cls(seq[0][0], n_components=MOG2_N_COMPONENTS, conservative=True)
        for frame, _ in seq:
            mask, _ = m.step(to_planar(frame))
        mask = np.asarray(mask)
        assert (mask[BOX] == 255).mean() > 0.9, (
            f"{cls.__name__} lost the stationary subject — its conservative "
            "branch is not doing anything")
        assert np.array_equal(mask, spec_mask), (
            f"{cls.__name__} disagrees with the spec under conservative update")


def test_a_global_exposure_change_does_not_latch_the_frame():
    """Protection released only by ordinary classification freezes for ever.

    A uniform brightness step makes most of the frame foreground at once; each
    of those pixels then freezes at the old exposure, and release would need
    `dist2 < Tb*var` with `var <= var_max`, which a shift beyond ~20 levels per
    channel can never satisfy. Measured on highway before the fix: 83% of the
    frame still foreground 700 frames later, against 14% for plain MOG2.

    Two mechanisms answer it and this test covers both: the per-pixel exit
    threshold `MOG2_PROTECT_EXIT`, and the frame-wide backstop
    `MOG2Base.CONSERVATIVE_MAX_COVERAGE`.
    """
    rng = np.random.default_rng(7)
    bg = rng.integers(40, 90, (H, W, 3), dtype=np.uint8)

    def coverage_after_step(conservative, protect_exit, max_coverage):
        model = GMM_CPU_NUMBA(bg, n_components=MOG2_N_COMPONENTS,
                              conservative=conservative, protect_exit=protect_exit)
        model.CONSERVATIVE_MAX_COVERAGE = max_coverage
        for i in range(260):
            f = bg.astype(np.float32) + (30.0 if i >= 60 else 0.0)
            f = np.clip(f + rng.normal(0, 2.5, f.shape), 0, 255).astype(np.uint8)
            mask, _ = model.step(to_planar(f))
        return (np.asarray(mask) == 255).mean()

    plain = coverage_after_step(False, 64.0, 0.5)
    broken = coverage_after_step(True, 0.0, 1.1)      # neither mechanism
    fixed = coverage_after_step(True, 64.0, 0.5)

    assert broken > 0.5, (
        "the failure this guards against did not reproduce — if the rule "
        f"changed, re-derive the guard rather than deleting it (got {broken:.2f})")
    assert fixed < 0.2, f"the frame is still latched: {fixed:.2f} foreground"
    assert fixed < broken / 2
    assert plain < 0.2
