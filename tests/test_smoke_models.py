"""Every model in `main.py`'s registry must run through the shared step() call.

This is the test that would have caught the interface breaks we hit before it
existed: `main.py` calling an undefined `step_func`, a step() signature drifting
between backends, and `video_gmm.py` calling `step(frame)` with one argument.
It exercises the exact call shape `main.py` uses, for every registered model.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pytest

from gmm import GMM_CPU, GMM_CPU_NUMBA, GMM_CUDA, GMM_CUPY

H, W, K = 32, 40, 5
UPDATE_ALPHA = np.float32(0.01)

CPU_MODELS = [GMM_CPU, GMM_CPU_NUMBA]
GPU_MODELS = [GMM_CUPY, GMM_CUDA]
CUPY_MODELS = (GMM_CUPY,)


def model_id(cls):
    """A GPU backend is None when its toolkit or device is missing, and None has
    no __name__ — which used to break collection rather than skip the test."""
    return cls.__name__ if cls is not None else "unavailable"


def cupy_is_mocked():
    """conftest swaps in a MagicMock for cupy so the module tree imports here.

    A CuPy model driven against that mock does not raise — it returns mock
    objects that `np.asarray` turns into an empty array — so the skip below has
    to test for the mock directly rather than wait for an exception.
    """
    from unittest.mock import MagicMock
    import cupy
    return isinstance(cupy, MagicMock)


def frames(n=4, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = np.full((H, W, 3), 110, np.uint8)
        f[8:20, 6 + i:18 + i] = 235
        out.append(np.clip(f.astype(np.int16) + rng.integers(-4, 5, f.shape),
                           0, 255).astype(np.uint8))
    return out


def drive(model_cls, seq):
    """Construct and step exactly the way main.py does."""
    model = model_cls(seq[0], n_components=K, parallel=True)
    masks = []
    for f in seq:
        planar = f.transpose(2, 0, 1).astype(np.float32)
        mask, cost = model.step(planar, UPDATE_ALPHA)
        masks.append(np.asarray(mask))
        assert cost >= 0, f"{model_cls.__name__}: step() returned a negative cost"
    return masks


@pytest.mark.parametrize('model_cls', CPU_MODELS, ids=model_id)
def test_model_runs(model_cls):
    seq = frames()
    masks = drive(model_cls, seq)
    for i, m in enumerate(masks):
        assert m.shape == (H, W), f"{model_cls.__name__}: mask shape {m.shape}"
        assert m.dtype == np.uint8, f"{model_cls.__name__}: mask dtype {m.dtype}"
        assert set(np.unique(m)) <= {0, 127, 255}, \
            f"{model_cls.__name__}: unexpected mask values {np.unique(m)}"


@pytest.mark.parametrize('model_cls', CPU_MODELS, ids=model_id)
def test_mask_is_not_degenerate(model_cls):
    """A backend that calls every pixel foreground (or background) is broken.

    The since-removed Stauffer-Grimson Numba backend once returned an
    all-foreground mask because its background loop broke on the cumulative
    weight before ever testing a match; this guards every current backend
    against that class of bug.
    """
    seq = frames(n=6)
    fg = (drive(model_cls, seq)[-1] == 255).mean()
    print(f"  {model_cls.__name__}: foreground = {fg * 100:.1f}%")
    assert 0.0 < fg < 0.9, f"{model_cls.__name__}: degenerate mask, fg={fg:.3f}"


@pytest.mark.parametrize('model_cls', GPU_MODELS, ids=model_id)
def test_gpu_model_runs(model_cls):
    if model_cls is None:
        pytest.skip("GPU model unavailable — cupy / numba.cuda not installed")
    if model_cls in CUPY_MODELS and cupy_is_mocked():
        pytest.skip(f"{model_cls.__name__}: cupy is mocked, masks are meaningless")
    try:
        masks = drive(model_cls, frames())
    except Exception as e:                       # pragma: no cover - no real GPU
        pytest.skip(f"{model_cls.__name__} unavailable on this machine: {e}")
    for m in masks:
        assert m.shape == (H, W)
        assert set(np.unique(m)) <= {0, 127, 255}


def test_step_signature_is_shared():
    """Every model must accept both `step(frame)` and `step(frame, alpha)`.

    step() was reduced from five parameters to two in fd847ab; callers across
    main.py, pipeline.py, benchmark.py and debug.py all went stale at once. This
    pins the surviving shape so the next change breaks here first.
    """
    seq = frames(n=2)
    planar = seq[1].transpose(2, 0, 1).astype(np.float32)
    for model_cls in CPU_MODELS:
        for args in ((), (UPDATE_ALPHA,)):
            model = model_cls(seq[0], n_components=K, parallel=True)
            mask, cost = model.step(planar, *args)
            assert mask.shape == (H, W), f"{model_cls.__name__} with {len(args)} arg(s)"
            assert cost >= 0, model_cls.__name__


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, '-v', '-s']))


@pytest.mark.parametrize('model_cls', CPU_MODELS, ids=model_id)
def test_bg_prob_carries_real_information(model_cls):
    """bg_prob is the seed the graph cut is built from, so an all-zero map is
    not a degraded result — it marks every pixel probable-foreground and the
    segmentation becomes noise. Nothing tested it: setting it to zero passed
    the whole suite.

    It is the summed weight of the modes that matched inside the background
    set, so it is high where the mask says background and exactly zero where
    the mask says foreground.
    """
    seq = frames(n=8)
    model = model_cls(seq[0], n_components=K, parallel=True)
    for f in seq:
        planar = f.transpose(2, 0, 1).astype(np.float32)
        model.step(planar, UPDATE_ALPHA)

    bg_prob = np.asarray(model.bg_prob)
    mask = np.asarray(model.mask)

    assert bg_prob.shape == (H, W) and bg_prob.dtype == np.float32
    assert bg_prob.min() >= 0.0 and bg_prob.max() <= 1.0, \
        f"outside [0,1]: [{bg_prob.min()}, {bg_prob.max()}]"
    assert bg_prob.any(), "bg_prob is identically zero — nothing is computing it"

    on_fg = bg_prob[mask == 255]
    on_bg = bg_prob[mask == 0]
    if on_fg.size:
        assert (on_fg == 0).all(), \
            "a foreground pixel matched no background mode, so its confidence must be 0"
    if on_bg.size:
        assert on_bg.mean() > 0.0, "background pixels must carry some confidence"


def test_bg_prob_agrees_across_backends():
    """The spec and the JIT must produce the same confidence, not just the same
    mask — the graph cut reads the confidence."""
    seq = frames(n=8)
    out = {}
    for cls in CPU_MODELS:
        m = cls(seq[0], n_components=K, parallel=True)
        for f in seq:
            m.step(f.transpose(2, 0, 1).astype(np.float32), UPDATE_ALPHA)
        out[cls.__name__] = np.asarray(m.bg_prob).copy()
    names = list(out)
    for other in names[1:]:
        assert np.allclose(out[names[0]], out[other], atol=1e-6), \
            f"{names[0]} and {other} disagree on bg_prob"
