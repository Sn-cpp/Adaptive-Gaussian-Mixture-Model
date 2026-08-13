"""Correctness suite for the MOG2 models.

Run directly (`python tests/test_mog2_correctness.py`) or under pytest.
Set NUMBA_ENABLE_CUDASIM=1 to exercise the CUDA kernels without a GPU.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2
import numpy as np
import pytest

from settings import (MOG2_BACKGROUND_RATIO, MOG2_HISTORY, MOG2_N_COMPONENTS,
                      MOG2_VAR_MAX, MOG2_VAR_MIN)
from gmm.mog2_common import opencv_reference, to_planar
from gmm.cpu.GMM_cpu import GMM_CPU
from gmm.cpu.GMM_cpu_numba import GMM_CPU_NUMBA


# ------------------------------------------------------------------ fixtures

def synthetic_sequence(n=30, H=48, W=64, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = np.empty((H, W, 3), np.uint8)
        f[:, :] = (60, 90, 120)
        f[:, :, 1] = np.tile(np.linspace(30, 200, W), (H, 1)).astype(np.uint8)
        cx = (5 + i) % max(W - 14, 1)
        f[H // 4:3 * H // 4, cx:cx + 12] = (240, 20, 20)
        f = np.clip(f.astype(np.int16) + rng.integers(-3, 4, f.shape), 0, 255)
        frames.append(f.astype(np.uint8))
    return frames


def run_model(model_cls, frames, color, **kw):
    model = model_cls(frames[0], n_components=MOG2_N_COMPONENTS, color=color, **kw)
    masks = [model.step(to_planar(f, color))[0].copy() for f in frames]
    if hasattr(model, 'sync_state'):
        model.sync_state()
    return masks, model


def iou(a, b):
    a, b = a == 255, b == 255
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum()) / float(u)


def prf(pred, gt):
    p, g = pred == 255, gt == 255
    if not p.any() and not g.any():
        return 1.0, 1.0, 1.0
    tp = float((p & g).sum())
    prec = tp / max(p.sum(), 1)
    rec = tp / max(g.sum(), 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


# ------------------------------------------------------------- OpenCV parity

def _opencv_parity(color, model_cls):
    frames = synthetic_sequence()
    mog2 = opencv_reference()
    # OpenCV takes BGR for the colour model, grayscale for the 1-channel one
    src = frames if color else [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    cv_masks = [mog2.apply(s) for s in src]
    our_masks, model = run_model(model_cls, frames, color)

    accs = [float(np.mean(a == b)) for a, b in zip(cv_masks, our_masks)]
    ious = [iou(a, b) for a, b in zip(cv_masks, our_masks)]
    prec, rec, f1 = prf(our_masks[-1], cv_masks[-1])
    print(f"  {'colour' if color else 'gray  '}  pixel-acc={np.mean(accs):.5f} "
          f"IoU={np.mean(ious):.5f} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    assert np.mean(ious) > 0.98, f"IoU {np.mean(ious):.4f} <= 0.98"
    assert np.mean(accs) > 0.999, f"pixel accuracy {np.mean(accs):.5f}"

    bg_ours = model.background_image()
    bg_cv = mog2.getBackgroundImage()
    if bg_cv is not None:
        bg_cv = bg_cv.reshape(bg_ours.shape)
        err = np.abs(bg_cv.astype(int) - bg_ours.astype(int)).max()
        print(f"          background image max abs error = {err}")
        assert err <= 1, f"background image differs by {err}"


def test_opencv_parity_gray():
    print("OpenCV MOG2 parity:")
    _opencv_parity(color=False, model_cls=GMM_CPU_NUMBA)


def test_opencv_parity_color():
    _opencv_parity(color=True, model_cls=GMM_CPU_NUMBA)


def test_opencv_parity_real_video():
    """Real 8-bit video, where OpenCV's FMA contraction can show up.

    Platform-dependent: on x86-64 Linux (Colab) this is exact — 0 of 2,304,000
    pixels differ. On macOS arm64 ~0.002% differ, because that OpenCV build
    contracts `acc += d*d` into an FMA and rounds once where we round twice.
    Budget: 0.05% of pixels.
    """
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    path = os.path.join(root, 'input.mp4')
    if not os.path.exists(path):
        pytest.skip("input.mp4 not found")

    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < 30:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (320, 240)))
    cap.release()
    if len(frames) < 10:
        pytest.skip("video too short")

    mog2 = opencv_reference()
    cv_masks = [mog2.apply(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
    our_masks, _ = run_model(GMM_CPU_NUMBA, frames, color=False)

    total = sum(m.size for m in cv_masks)
    ndiff = sum(int((a != b).sum()) for a, b in zip(cv_masks, our_masks))
    inter = sum(int(((a == 255) & (b == 255)).sum()) for a, b in zip(cv_masks, our_masks))
    union = sum(int(((a == 255) | (b == 255)).sum()) for a, b in zip(cv_masks, our_masks))
    ratio = ndiff / total
    print(f"  real video: {ndiff}/{total} px differ ({ratio * 100:.5f}%), "
          f"IoU={inter / max(union, 1):.5f}")
    assert ratio < 5e-4, f"{ratio * 100:.4f}% of pixels differ from OpenCV"
    assert inter / max(union, 1) > 0.98


# ------------------------------------------------------------ model agreement

def _compare_models(a_cls, b_cls, label, n=12, H=32, W=40, color=False):
    frames = synthetic_sequence(n=n, H=H, W=W)
    m_a, model_a = run_model(a_cls, frames, color)
    m_b, model_b = run_model(b_cls, frames, color)
    for i, (x, y) in enumerate(zip(m_a, m_b)):
        assert np.array_equal(x, y), f"{label}: mask mismatch on frame {i}"
    for key in ('weights', 'means', 'vars'):
        err = np.abs(getattr(model_a, key) - getattr(model_b, key)).max()
        print(f"  {label}  max |{key}| error = {err:.3e}")
        assert err < 1e-5, f"{key} error {err}"
    assert np.array_equal(model_a.modes, model_b.modes)


def test_sequential_matches_numba():
    _compare_models(GMM_CPU, GMM_CPU_NUMBA, 'sequential vs numba')


def test_cuda_matches_cpu():
    simulator = os.environ.get('NUMBA_ENABLE_CUDASIM') == '1'
    try:
        from gmm.gpu.GMM_cuda import GMM_CUDA, is_available
        if not simulator and not is_available():
            pytest.skip("CUDA unavailable")
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"CUDA import failed: {e}")
    n, H, W = (3, 8, 8) if simulator else (12, 48, 64)
    _compare_models(GMM_CPU_NUMBA, GMM_CUDA, 'cpu vs cuda', n, H, W)


def _cupy_usable():
    """conftest may substitute a MagicMock for cupy; a mock swallows every call
    without raising, so an exception-based probe is not enough."""
    from unittest.mock import MagicMock
    try:
        import cupy as cp
        if isinstance(cp, MagicMock):
            return False
        cp.zeros(1)
        return True
    except Exception:                                        # pragma: no cover
        return False


def test_cupy_matches_cpu():
    """The CuPy RawKernel backend against the plain-Python reference.

    Measured on a Colab T4: bit-exact, so this asserts equality.
    """
    if not _cupy_usable():
        pytest.skip("CuPy unavailable")
    from gmm.gpu.GMM_cupy import GMM_CUPY
    _compare_models(GMM_CPU, GMM_CUPY, 'cpu vs cupy')


def test_cupy_matches_numba_cuda():
    """The two GPU backends implement one algorithm through two toolchains
    (CuPy RawKernel vs numba.cuda); a disagreement is a defect in one of them."""
    if not _cupy_usable():
        pytest.skip("CuPy unavailable")
    try:
        from gmm.gpu.GMM_cuda import GMM_CUDA, is_available
        if not is_available():
            pytest.skip("CUDA unavailable")
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"CUDA import failed: {e}")
    from gmm.gpu.GMM_cupy import GMM_CUPY
    _compare_models(GMM_CUDA, GMM_CUPY, 'numba.cuda vs cupy')


def test_full_pipeline_backends_agree():
    """Blur + morphology + composite, end to end."""
    from pipeline import make_pipeline
    frames = synthetic_sequence(n=6, H=32, W=40)
    outs = {}
    for name, cls in (('sequential', GMM_CPU), ('numba', GMM_CPU_NUMBA)):
        p = make_pipeline(cls, frames[0], n_components=MOG2_N_COMPONENTS)
        for f in frames:
            out, mask, _ = p.process(f)
        outs[name] = (out.copy(), mask.copy())
    d = np.abs(outs['sequential'][0].astype(int) - outs['numba'][0].astype(int)).max()
    print(f"  full pipeline max |output| difference = {d}")
    assert np.array_equal(outs['sequential'][1], outs['numba'][1])
    assert d <= 1, f"composite differs by {d}"


# ------------------------------------------------------- behavioural checks

def _absorption_frames(apply_fn, bg, obj, box, warm=40, limit=2000):
    for _ in range(warm):
        apply_fn(bg)
    n = 0
    for _ in range(limit):
        mask = apply_fn(obj)
        if (mask[box] == 255).mean() > 0.5:
            n += 1
        else:
            break
    return n


def test_stationary_object_persists():
    """A stopped object stays foreground exactly as long as it does in OpenCV.

    The absorption time is ln(TB)/ln(1-alpha) frames — the *old* background mode
    has to decay below the background ratio — not TB * history. At the default
    alpha = 1/500 that is ~53 frames.
    """
    H, W = 24, 32
    box = (slice(8, 16), slice(10, 22))
    bg = np.full((H, W, 3), 100, np.uint8)
    obj = bg.copy()
    obj[box] = 230

    for alpha in (0.02, 0.005, 1.0 / MOG2_HISTORY):
        mog2 = opencv_reference(detect_shadows=False)
        n_cv = _absorption_frames(
            lambda f: mog2.apply(f, learningRate=alpha), bg, obj, box)

        model = GMM_CPU_NUMBA(bg, n_components=MOG2_N_COMPONENTS,
                                   detect_shadows=False)
        n_ours = _absorption_frames(
            lambda f: model.step(to_planar(f), alpha)[0], bg, obj, box)

        theory = math.log(MOG2_BACKGROUND_RATIO) / math.log(1 - alpha)
        print(f"  alpha={alpha:.4f}: ours={n_ours} opencv={n_cv} "
              f"theory={theory:.1f} frames")
        assert n_ours == n_cv, f"absorption {n_ours} != OpenCV {n_cv}"
        assert abs(n_ours - theory) <= 3, "absorption time off the analytic value"

    # keep showing the object: it must eventually become *the* background
    for _ in range(2000):
        model.step(to_planar(obj), alpha)
    learned = model.background_image()[box].mean()
    print(f"  learned background inside the box = {learned:.1f} (object = 230)")
    assert learned > 200, f"background model did not absorb the object ({learned:.1f})"


def test_long_run_stability():
    n = int(os.environ.get('LONG_RUN_FRAMES', 1200))
    H, W = 24, 32
    rng = np.random.default_rng(7)
    base = np.zeros((H, W, 3), np.uint8)
    model = GMM_CPU_NUMBA(base, n_components=MOG2_N_COMPONENTS)
    for i in range(n):
        f = np.clip(rng.normal(120, 20, (H, W, 3)), 0, 255).astype(np.uint8)
        if i % 7 == 0:
            f[5:15, 5:20] = 250
        model.step(to_planar(f))

    w, m, v, modes = model.weights, model.means, model.vars, model.modes
    assert not np.isnan(w).any() and not np.isnan(m).any() and not np.isnan(v).any()
    assert (modes >= 1).all() and (modes <= MOG2_N_COMPONENTS).all(), "invalid mode count"
    assert (v >= 0).all(), "negative variance"
    active = np.arange(MOG2_N_COMPONENTS)[:, None, None] < modes[None]
    assert (v[active] >= MOG2_VAR_MIN - 1e-4).all(), "variance below varMin"
    assert (v[active] <= MOG2_VAR_MAX + 1e-4).all(), "variance above varMax"
    assert (w[active] >= 0).all(), "negative weight"
    # Active weights sum to 1, up to the mass MOG2 itself loses: a mode pruned
    # mid-traversal has already contributed to totalWeight but is excluded from
    # the renormalisation, so the sum can sit just below 1. OpenCV behaves the
    # same way; what matters is that it never drifts upward or runs away.
    wsum = np.where(active, w, 0.0).sum(axis=0)
    lo, hi = wsum.min(), wsum.max()
    print(f"  {n} frames: sum(active weights) in [{lo:.6f}, {hi:.6f}], "
          f"modes in [{modes.min()}, {modes.max()}]")
    assert hi <= 1.0 + 1e-4, f"weights sum above 1 ({hi})"
    assert lo > 0.95, f"weights decayed away (min sum {lo})"


TESTS = [test_opencv_parity_gray, test_opencv_parity_color,
         test_opencv_parity_real_video,
         test_sequential_matches_numba, test_cuda_matches_cpu,
         test_cupy_matches_cpu, test_cupy_matches_numba_cuda,
         test_full_pipeline_backends_agree,
         test_stationary_object_persists, test_long_run_stability]

if __name__ == "__main__":
    failed = 0
    skipped = 0
    for t in TESTS:
        print(f"\n=== {t.__name__}")
        try:
            t()
            print("  PASS")
        except pytest.skip.Exception as e:
            skipped += 1
            print(f"  SKIP: {e}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
    print(f"\n{len(TESTS) - failed - skipped}/{len(TESTS)} passed, "
          f"{skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)
