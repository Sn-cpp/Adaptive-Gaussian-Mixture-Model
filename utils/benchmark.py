"""Benchmark and mask-quality helpers used by the deliverable notebook."""
import time

import cv2
import numpy as np


def load_video(source, max_frames=None):
    cap = cv2.VideoCapture(source)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def resize_frames(frames, width, height):
    return [cv2.resize(f, (width, height)) for f in frames]


def synthetic_frames(n, H, W, seed=0):
    """Fallback input when no video file is available."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = np.empty((H, W, 3), np.uint8)
        f[:, :] = (60, 90, 120)
        f[:, :, 1] = np.tile(np.linspace(30, 200, W), (H, 1)).astype(np.uint8)
        cx = int((W - W // 6) * (0.5 + 0.45 * np.sin(i * 0.15)))
        f[H // 4:3 * H // 4, cx:cx + W // 6] = (240, 20, 20)
        out.append(np.clip(f.astype(np.int16) + rng.integers(-3, 4, f.shape),
                           0, 255).astype(np.uint8))
    return out


def benchmark(model_cls, frames, n_warmup=2, n_measure=5, **pipeline_kw):
    """Median per-frame timing for one model. Returns a flat dict."""
    from pipeline import make_pipeline
    H, W = frames[0].shape[:2]

    for _ in range(n_warmup):
        p = make_pipeline(model_cls, frames[0], **pipeline_kw)
        for f in frames[:min(3, len(frames))]:
            p.process(f)

    runs = []
    for _ in range(n_measure):
        p = make_pipeline(model_cls, frames[0], **pipeline_kw)
        runs.extend(p.process(f)[2] for f in frames)

    row = {'model': model_cls.__name__, 'resolution': f'{W}x{H}'}
    for key in runs[0]:
        row[key] = float(np.median([t[key] for t in runs]))
    row['fps'] = 1.0 / max(row['total'], 1e-9)
    return row


def benchmark_streamed(model_cls, frames, n_measure=3, **pipeline_kw):
    """Throughput of the pipelined CUDA path (upload overlapped with compute)."""
    from pipeline import make_pipeline
    H, W = frames[0].shape[:2]
    p = make_pipeline(model_cls, frames[0], **pipeline_kw)
    if not hasattr(p, 'process_stream'):
        return None
    for _ in p.process_stream(frames[:min(4, len(frames))]):
        pass
    best = None
    for _ in range(n_measure):
        p = make_pipeline(model_cls, frames[0], **pipeline_kw)
        t0 = time.perf_counter()
        for _ in p.process_stream(frames):
            pass
        dt = (time.perf_counter() - t0) / len(frames)
        best = dt if best is None else min(best, dt)
    return {'model': model_cls.__name__ + ' (streamed)', 'resolution': f'{W}x{H}',
            'total': best, 'fps': 1.0 / max(best, 1e-9)}


def benchmark_blur_variants(frames, n=5, use_cuda=False):
    """Separable (2 x 1D) vs naive 2D convolution."""
    from gmm.mog2_common import create_gaussian_kernel, create_gaussian_kernel_1d

    if use_cuda:
        from utils import blur_cuda
        return blur_cuda.blur_variants(frames[0], n)

    from utils import blur_numba
    blur_numba.warmup()
    H, W = frames[0].shape[:2]
    k1d = create_gaussian_kernel_1d()
    k2d = create_gaussian_kernel()
    tmp = np.zeros((H, W, 3), np.float32)
    out = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)

    def timed(fn):
        fn()                       # warm up / JIT
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    sep = timed(lambda: (blur_numba.blur_h(frames[0], tmp, k1d),
                         blur_numba.blur_v_composite(tmp, frames[0], mask, out, k1d)))
    naive = timed(lambda: blur_numba.blur_2d(frames[0], out, k2d))
    return {'separable_ms': sep * 1e3, 'naive_2d_ms': naive * 1e3,
            'speedup': naive / max(sep, 1e-9)}


def display_side_by_side(original, mask, output):
    return np.hstack([original, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), output])


def compute_iou(mask_a, mask_b):
    a, b = mask_a == 255, mask_b == 255
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum()) / float(union)


def mask_metrics(pred, gt):
    """Pixel accuracy / IoU / precision / recall / F1 against a reference mask."""
    p, g = pred == 255, gt == 255
    if not p.any() and not g.any():        # both empty -> perfect agreement
        return {'accuracy': float(np.mean(pred == gt)), 'iou': 1.0,
                'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    tp = float((p & g).sum())
    prec = tp / max(p.sum(), 1)
    rec = tp / max(g.sum(), 1)
    return {
        'accuracy': float(np.mean(pred == gt)),
        'iou': compute_iou(pred, gt),
        'precision': prec,
        'recall': rec,
        'f1': 2 * prec * rec / max(prec + rec, 1e-9),
    }
