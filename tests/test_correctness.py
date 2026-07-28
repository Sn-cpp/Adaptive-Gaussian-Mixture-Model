"""
Correctness tests for Adaptive Gaussian Mixture Model backends.

Compares GMM_CPU (NumPy vectorized) against GMM_CPU_NUMBA (serial & parallel),
and optionally against GMM_CUPY_V0 / GMM_CUPY_V1 when CuPy is available.

Authors: Hai Duong Huynh Le (22127081), Duc Tin (22127415)
"""

import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import cupy
    _CUPY_REAL = not isinstance(cupy, MagicMock)
except ImportError:
    _CUPY_REAL = False

import numpy as np
import pytest

from gmm import GMM_CPU, GMM_CPU_NUMBA

requires_cupy = pytest.mark.skipif(
    not _CUPY_REAL,
    reason="CuPy/CUDA not available on this platform"
)

MASK_TOL_CPU = 0.01
MASK_TOL_NUMBA_PAR = 0.02
MASK_TOL_GPU = 0.03


def make_test_frames(H, W, n_frames=10, seed=42):
    rng = np.random.RandomState(seed)
    bg = rng.randint(100, 160, (H, W, 3), dtype=np.uint8)
    frames = []
    for i in range(n_frames):
        frame = bg.copy()
        noise = rng.randint(-5, 6, (H, W, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        y0, x0 = H // 4, W // 4
        obj_h, obj_w = H // 4, W // 4
        frame[y0:y0 + obj_h, x0:x0 + obj_w] = rng.randint(
            0, 256, (obj_h, obj_w, 3), dtype=np.uint8
        )
        frames.append(frame)
    return frames


def to_planar(frame):
    return frame.transpose(2, 0, 1).astype(np.float32)


def compare_masks(mask_a, mask_b, tolerance):
    diff = np.count_nonzero(mask_a != mask_b) / mask_a.size
    return diff <= tolerance, diff


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:

    def test_state_shapes(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        first = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        model = GMM_CPU(first, n_components=K)
        assert model.means.shape == (K, 3, H, W)
        assert model.vars.shape == (K, H, W)
        assert model.weights.shape == (K, H, W)

    def test_initial_weights(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        first = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        model = GMM_CPU(first, n_components=K)
        assert np.allclose(model.weights[0], 1.0)
        assert np.allclose(model.weights[1:], 0.0)

    def test_initial_variance(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        first = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        model = GMM_CPU(first, n_components=K)
        from settings import INIT_VAR
        assert np.allclose(model.vars, INIT_VAR)

    def test_first_mean_matches_frame(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        first = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        model = GMM_CPU(first, n_components=K)
        expected = first.transpose(2, 0, 1).astype(np.float32)
        assert np.allclose(model.means[0], expected)

    def test_numba_init_matches_cpu(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        first = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        cpu = GMM_CPU(first.copy(), n_components=K)
        numba = GMM_CPU_NUMBA(first.copy(), n_components=K, parallel=False)
        assert np.allclose(cpu.means, numba.means)
        assert np.allclose(cpu.vars, numba.vars)
        assert np.allclose(cpu.weights, numba.weights)


# ---------------------------------------------------------------------------
# CPU (NumPy) vs Numba serial
# ---------------------------------------------------------------------------

class TestCpuVsNumbaSerial:

    def test_single_frame_mask(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']

        first = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        cpu = GMM_CPU(first.copy(), n_components=K)
        numba = GMM_CPU_NUMBA(first.copy(), n_components=K, parallel=False)

        frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        fp = to_planar(frame)

        mask_cpu, _ = cpu.predict(fp.copy(), mt, bt)
        mask_numba, _ = numba.predict(fp.copy(), mt, bt)

        ok, diff = compare_masks(mask_cpu, mask_numba, MASK_TOL_CPU)
        assert ok, f"Mask diff {diff:.4f} exceeds {MASK_TOL_CPU}"

    def test_multi_frame_consistency(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=10)
        cpu = GMM_CPU(frames[0].copy(), n_components=K)
        numba = GMM_CPU_NUMBA(frames[0].copy(), n_components=K, parallel=False)

        for frame in frames[1:]:
            fp = to_planar(frame)
            mask_cpu, dsq_cpu = cpu.predict(fp.copy(), mt, bt)
            cpu.update(fp.copy(), dsq_cpu, mt, alpha)
            mask_numba, dsq_numba = numba.predict(fp.copy(), mt, bt)
            numba.update(fp.copy(), dsq_numba, mt, alpha)

        ok, diff = compare_masks(mask_cpu, mask_numba, MASK_TOL_CPU)
        assert ok, f"After 10 frames, mask diff {diff:.4f} exceeds {MASK_TOL_CPU}"

    def test_weights_normalized_cpu(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=5)
        model = GMM_CPU(frames[0].copy(), n_components=K)

        for frame in frames[1:]:
            fp = to_planar(frame)
            mask, dsq = model.predict(fp.copy(), mt, bt)
            model.update(fp.copy(), dsq, mt, alpha)

        sums = model.weights.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-5), \
            f"Weight sums deviate: max={sums.max():.6f}, min={sums.min():.6f}"

    def test_weights_normalized_numba(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=5)
        model = GMM_CPU_NUMBA(frames[0].copy(), n_components=K, parallel=False)

        for frame in frames[1:]:
            fp = to_planar(frame)
            mask, dsq = model.predict(fp.copy(), mt, bt)
            model.update(fp.copy(), dsq, mt, alpha)

        sums = model.weights.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-5), \
            f"Numba weight sums deviate: max={sums.max():.6f}, min={sums.min():.6f}"


# ---------------------------------------------------------------------------
# CPU (NumPy) vs Numba parallel
# ---------------------------------------------------------------------------

class TestCpuVsNumbaParallel:

    def test_single_frame_mask(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']

        first = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        cpu = GMM_CPU(first.copy(), n_components=K)
        numba_par = GMM_CPU_NUMBA(first.copy(), n_components=K, parallel=True)

        frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        fp = to_planar(frame)

        mask_cpu, _ = cpu.predict(fp.copy(), mt, bt)
        mask_par, _ = numba_par.predict(fp.copy(), mt, bt)

        ok, diff = compare_masks(mask_cpu, mask_par, MASK_TOL_NUMBA_PAR)
        assert ok, f"Mask diff {diff:.4f} exceeds {MASK_TOL_NUMBA_PAR}"

    def test_multi_frame_consistency(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=10)
        cpu = GMM_CPU(frames[0].copy(), n_components=K)
        numba_par = GMM_CPU_NUMBA(frames[0].copy(), n_components=K, parallel=True)

        for frame in frames[1:]:
            fp = to_planar(frame)
            mask_cpu, dsq_cpu = cpu.predict(fp.copy(), mt, bt)
            cpu.update(fp.copy(), dsq_cpu, mt, alpha)
            mask_par, dsq_par = numba_par.predict(fp.copy(), mt, bt)
            numba_par.update(fp.copy(), dsq_par, mt, alpha)

        ok, diff = compare_masks(mask_cpu, mask_par, MASK_TOL_NUMBA_PAR)
        assert ok, f"After 10 frames, mask diff {diff:.4f} exceeds {MASK_TOL_NUMBA_PAR}"

    def test_numba_serial_vs_parallel(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=5)
        serial = GMM_CPU_NUMBA(frames[0].copy(), n_components=K, parallel=False)
        par = GMM_CPU_NUMBA(frames[0].copy(), n_components=K, parallel=True)

        for frame in frames[1:]:
            fp = to_planar(frame)
            mask_s, dsq_s = serial.predict(fp.copy(), mt, bt)
            serial.update(fp.copy(), dsq_s, mt, alpha)
            mask_p, dsq_p = par.predict(fp.copy(), mt, bt)
            par.update(fp.copy(), dsq_p, mt, alpha)

        ok, diff = compare_masks(mask_s, mask_p, MASK_TOL_CPU)
        assert ok, f"Serial vs parallel Numba diff {diff:.4f} exceeds {MASK_TOL_CPU}"

    def test_weights_normalized_parallel(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=5)
        model = GMM_CPU_NUMBA(frames[0].copy(), n_components=K, parallel=True)

        for frame in frames[1:]:
            fp = to_planar(frame)
            mask, dsq = model.predict(fp.copy(), mt, bt)
            model.update(fp.copy(), dsq, mt, alpha)

        sums = model.weights.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-5), \
            f"Parallel weight sums deviate: max={sums.max():.6f}, min={sums.min():.6f}"


# ---------------------------------------------------------------------------
# GPU (CuPy) backends
# ---------------------------------------------------------------------------

@requires_cupy
class TestGpuCupyV0:

    def test_mask_vs_cpu(self, small_dims, default_params):
        import cupy as cp
        from gmm import GMM_CUPY_V0

        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=5)
        cpu = GMM_CPU(frames[0].copy(), n_components=K)
        gpu = GMM_CUPY_V0(frames[0].copy(), n_components=K)

        for frame in frames[1:]:
            fp = to_planar(frame)
            mask_cpu, dsq_cpu = cpu.predict(fp.copy(), mt, bt)
            cpu.update(fp.copy(), dsq_cpu, mt, alpha)
            fp_gpu = cp.asarray(fp)
            mask_gpu, dsq_gpu = gpu.predict(fp_gpu, mt, bt)
            gpu.update(fp_gpu, dsq_gpu, mt, alpha)

        mask_gpu_np = cp.asnumpy(mask_gpu)
        ok, diff = compare_masks(mask_cpu, mask_gpu_np, MASK_TOL_GPU)
        assert ok, f"CuPy V0 vs CPU diff {diff:.4f} exceeds {MASK_TOL_GPU}"

    def test_weights_normalized(self, small_dims, default_params):
        import cupy as cp
        from gmm import GMM_CUPY_V0

        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=5)
        model = GMM_CUPY_V0(frames[0].copy(), n_components=K)

        for frame in frames[1:]:
            fp_gpu = cp.asarray(to_planar(frame))
            mask, dsq = model.predict(fp_gpu, mt, bt)
            model.update(fp_gpu, dsq, mt, alpha)

        sums = cp.asnumpy(model.weights.sum(axis=0))
        assert np.allclose(sums, 1.0, atol=1e-4), "CuPy V0 weight sums deviate"


@requires_cupy
class TestGpuCupyV1:

    def test_mask_vs_cpu(self, small_dims, default_params):
        import cupy as cp
        from gmm import GMM_CUPY_V1

        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=5)
        cpu = GMM_CPU(frames[0].copy(), n_components=K)
        gpu = GMM_CUPY_V1(frames[0].copy(), n_components=K)

        for frame in frames[1:]:
            fp = to_planar(frame)
            mask_cpu, dsq_cpu = cpu.predict(fp.copy(), mt, bt)
            cpu.update(fp.copy(), dsq_cpu, mt, alpha)
            fp_gpu = cp.asarray(fp)
            mask_gpu, dsq_gpu = gpu.predict(fp_gpu, mt, bt)
            gpu.update(fp_gpu, dsq_gpu, mt, alpha)

        mask_gpu_np = cp.asnumpy(mask_gpu)
        ok, diff = compare_masks(mask_cpu, mask_gpu_np, MASK_TOL_GPU)
        assert ok, f"CuPy V1 vs CPU diff {diff:.4f} exceeds {MASK_TOL_GPU}"

    def test_v0_vs_v1(self, small_dims, default_params):
        import cupy as cp
        from gmm import GMM_CUPY_V0, GMM_CUPY_V1

        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=5)
        v0 = GMM_CUPY_V0(frames[0].copy(), n_components=K)
        v1 = GMM_CUPY_V1(frames[0].copy(), n_components=K)

        for frame in frames[1:]:
            fp_gpu = cp.asarray(to_planar(frame))
            mask_v0, dsq_v0 = v0.predict(fp_gpu.copy(), mt, bt)
            v0.update(fp_gpu.copy(), dsq_v0, mt, alpha)
            mask_v1, dsq_v1 = v1.predict(fp_gpu.copy(), mt, bt)
            v1.update(fp_gpu.copy(), dsq_v1, mt, alpha)

        mask_v0_np = cp.asnumpy(mask_v0)
        mask_v1_np = cp.asnumpy(mask_v1)
        ok, diff = compare_masks(mask_v0_np, mask_v1_np, MASK_TOL_GPU)
        assert ok, f"CuPy V0 vs V1 diff {diff:.4f} exceeds {MASK_TOL_GPU}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_uniform_frame_all_background(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']

        first = np.full((H, W, 3), 128, dtype=np.uint8)
        model = GMM_CPU(first.copy(), n_components=K)

        fp = to_planar(first)
        mask, _ = model.predict(fp, mt, bt)

        fg_ratio = np.count_nonzero(mask) / mask.size
        assert fg_ratio < 0.01, \
            f"Uniform frame: {fg_ratio:.4f} foreground (expected <1%)"

    def test_all_different_mostly_foreground(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']

        first = np.zeros((H, W, 3), dtype=np.uint8)
        model = GMM_CPU(first.copy(), n_components=K)

        different = np.full((H, W, 3), 255, dtype=np.uint8)
        fp = to_planar(different)
        mask, _ = model.predict(fp, mt, bt)

        fg_ratio = np.count_nonzero(mask) / mask.size
        assert fg_ratio > 0.9, \
            f"All-different frame: {fg_ratio:.4f} foreground (expected >90%)"

    def test_single_pixel(self, default_params):
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        first = np.array([[[100, 150, 200]]], dtype=np.uint8)
        model = GMM_CPU(first.copy(), n_components=K)

        fp = to_planar(first)
        mask, dsq = model.predict(fp, mt, bt)
        model.update(fp, dsq, mt, alpha)

        assert mask.shape == (1, 1)
        assert model.weights.shape == (K, 1, 1)

    def test_gradual_background_learning(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        first = np.full((H, W, 3), 100, dtype=np.uint8)
        model = GMM_CPU(first.copy(), n_components=K)

        static = np.full((H, W, 3), 100, dtype=np.uint8)
        fp = to_planar(static)
        for _ in range(30):
            mask, dsq = model.predict(fp.copy(), mt, bt)
            model.update(fp.copy(), dsq, mt, alpha)

        fg_ratio = np.count_nonzero(mask) / mask.size
        assert fg_ratio < 0.01, \
            f"After 30 static frames: {fg_ratio:.4f} foreground (expected <1%)"


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

class TestPostProcessing:

    def test_mask_refiner_preserves_shape(self, small_dims):
        from utils.post_processing import mask_refiner
        H, W = small_dims
        mask = np.random.randint(0, 2, (H, W), dtype=np.uint8) * 255
        refined = mask_refiner(mask)
        assert refined.shape == mask.shape
        assert refined.dtype == np.uint8

    def test_background_subtractor_preserves_shape(self, small_dims):
        from utils.post_processing import background_subtractor
        H, W = small_dims
        frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (H, W), dtype=np.uint8) * 255
        result = background_subtractor(frame, mask)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Per-frame divergence tracking
# ---------------------------------------------------------------------------

class TestPerFrameDivergence:

    def test_cpu_vs_numba_divergence_bounded(self, small_dims, default_params):
        H, W = small_dims
        K = default_params['n_components']
        mt = default_params['match_threshold']
        bt = default_params['bg_threshold']
        alpha = default_params['alpha']

        frames = make_test_frames(H, W, n_frames=20)
        cpu = GMM_CPU(frames[0].copy(), n_components=K)
        numba = GMM_CPU_NUMBA(frames[0].copy(), n_components=K, parallel=False)

        max_diff = 0.0
        for frame in frames[1:]:
            fp = to_planar(frame)
            mask_cpu, dsq_cpu = cpu.predict(fp.copy(), mt, bt)
            cpu.update(fp.copy(), dsq_cpu, mt, alpha)
            mask_numba, dsq_numba = numba.predict(fp.copy(), mt, bt)
            numba.update(fp.copy(), dsq_numba, mt, alpha)
            _, diff = compare_masks(mask_cpu, mask_numba, 1.0)
            max_diff = max(max_diff, diff)

        assert max_diff <= MASK_TOL_CPU, \
            f"Max divergence {max_diff:.4f} over 20 frames exceeds {MASK_TOL_CPU}"
