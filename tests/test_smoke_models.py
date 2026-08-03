"""Every model in `main.py`'s registry must run through the shared step() call.

This is the test that would have caught the three interface breaks we hit:
`main.py` calling an undefined `step_func`, `GMM_CPU_NUMBA.step` requiring a
fifth argument nobody passed, and `video_gmm.py` calling `step(frame)` with one.
It exercises the exact call shape `main.py` uses, for every registered model.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pytest

from gmm import (GMM_CPU, GMM_CPU_NUMBA, GMM_CPU_MOG2, GMM_CPU_NUMBA_MOG2,
                 GMM_CUDA_MOG2, GMM_CUPY_V0, GMM_CUPY_V1)

H, W, K = 32, 40, 5
MATCH_THRESHOLD = np.float32(3.5)
UPDATE_ALPHA = np.float32(0.01)
WEIGHT_THRESHOLD = np.float32(0.7)

CPU_MODELS = [GMM_CPU, GMM_CPU_NUMBA, GMM_CPU_MOG2, GMM_CPU_NUMBA_MOG2]
GPU_MODELS = [GMM_CUPY_V0, GMM_CUPY_V1, GMM_CUDA_MOG2]


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
        mask, cost = model.step(planar, MATCH_THRESHOLD, UPDATE_ALPHA,
                                WEIGHT_THRESHOLD)
        masks.append(np.asarray(mask))
        assert cost >= 0, f"{model_cls.__name__}: step() returned a negative cost"
    return masks


@pytest.mark.parametrize('model_cls', CPU_MODELS, ids=lambda c: c.__name__)
def test_model_runs(model_cls):
    seq = frames()
    masks = drive(model_cls, seq)
    for i, m in enumerate(masks):
        assert m.shape == (H, W), f"{model_cls.__name__}: mask shape {m.shape}"
        assert m.dtype == np.uint8, f"{model_cls.__name__}: mask dtype {m.dtype}"
        assert set(np.unique(m)) <= {0, 127, 255}, \
            f"{model_cls.__name__}: unexpected mask values {np.unique(m)}"


@pytest.mark.parametrize('model_cls', CPU_MODELS, ids=lambda c: c.__name__)
def test_mask_is_not_degenerate(model_cls):
    """A backend that calls every pixel foreground (or background) is broken.

    GMM_CPU_NUMBA used to return an all-foreground mask because its background
    loop broke on the cumulative weight before ever testing a match.
    """
    seq = frames(n=6)
    fg = (drive(model_cls, seq)[-1] == 255).mean()
    print(f"  {model_cls.__name__}: foreground = {fg * 100:.1f}%")
    assert 0.0 < fg < 0.9, f"{model_cls.__name__}: degenerate mask, fg={fg:.3f}"


@pytest.mark.parametrize('model_cls', GPU_MODELS, ids=lambda c: c.__name__)
def test_gpu_model_runs(model_cls):
    if model_cls is None:
        pytest.skip("GPU model unavailable — cupy / numba.cuda not installed")
    try:
        masks = drive(model_cls, frames())
    except Exception as e:                       # pragma: no cover - no real GPU
        pytest.skip(f"{model_cls.__name__} unavailable on this machine: {e}")
    for m in masks:
        assert m.shape == (H, W)
        assert set(np.unique(m)) <= {0, 127, 255}


def test_step_accepts_the_optional_fifth_argument():
    """debug.py passes comp_gen_threshold; every model must tolerate it."""
    seq = frames(n=2)
    for model_cls in CPU_MODELS:
        model = model_cls(seq[0], n_components=K, parallel=True)
        planar = seq[1].transpose(2, 0, 1).astype(np.float32)
        mask, _ = model.step(planar, MATCH_THRESHOLD, UPDATE_ALPHA,
                             WEIGHT_THRESHOLD, np.float32(9.0))
        assert mask.shape == (H, W), model_cls.__name__


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, '-v', '-s']))
