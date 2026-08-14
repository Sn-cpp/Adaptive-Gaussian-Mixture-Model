"""Per-stage timing for the three post-processing versions. Needs a real GPU.

    python bench_post.py --sizes 480 720 1080

    v0  mask on the device, threshold + median + fill on the host with OpenCV
    v1  threshold + median as CUDA kernels, fill on the host
    v2  threshold fused into the model kernel's epilogue, median tiled

The number that matters is not the kernel time — it is the mask round trip. v0
copies the confidence map back (float32, 4 bytes a pixel) so the host can
threshold it; v1 and v2 copy back one byte a pixel, already refined. At 1080p
that is 8 MB against 2 MB per frame.

Timings are medians of interleaved repeats. Interleave if you re-measure: a
single cold pass on a shared T4 has reported swings of 30% and more, which is
larger than every effect this file is trying to measure.
"""
import argparse
import time

import cv2
import numpy as np

from gmm_mask import GMM_Mask_CUDA, GMM_Mask_CUDA_v1, GMM_Mask_CUDA_v2
from settings import MOG2_BG_PROB_THRESHOLD
from utils.post_processing import fill_holes, refine_mask

SIZES = {480: (854, 480), 720: (1280, 720), 1080: (1920, 1080)}


def make_frames(n, size, seed=0):
    """Synthetic traffic: a static textured road with blobs moving across it."""
    w, h = size
    rng = np.random.default_rng(seed)
    road = cv2.GaussianBlur(
        rng.integers(60, 110, (h, w, 3), dtype=np.uint8), (7, 7), 0)
    out = []
    for i in range(n):
        f = road.astype(np.float32)
        for k in range(4):
            cx = int((i * 7 + k * w // 4) % w)
            cy = h // 2 + (k - 2) * h // 8
            cv2.rectangle(f, (cx, cy), (cx + w // 12, cy + h // 14),
                          (200.0, 195.0, 205.0), -1)
        out.append(np.ascontiguousarray(
            np.clip(f + rng.normal(0, 2.0, f.shape), 0, 255), dtype=np.float32))
    return out


def timed(fn, frames, warm=8, repeats=3):
    from numba import cuda
    for f in frames[:warm]:
        fn(f)
    cuda.synchronize()
    best = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for f in frames[warm:]:
            fn(f)
        cuda.synchronize()
        best.append((time.perf_counter() - t0) / len(frames[warm:]))
    return np.median(best) * 1000


def bench(size, n=40):
    w, h = size
    frames = make_frames(n, size)

    def v0(f):
        mask, bg_prob, _ = m0.apply(f)
        return refine_mask(np.asarray(mask), bg_prob=np.asarray(bg_prob))

    def vgpu(model):
        def run(f):
            mask, _, _ = model.apply(f)
            return fill_holes(np.asarray(mask))
        return run

    m0 = GMM_Mask_CUDA(h, w)
    m1 = GMM_Mask_CUDA_v1(h, w)
    m2 = GMM_Mask_CUDA_v2(h, w)

    rows = []
    # interleave so thermal drift and host contention hit all three equally
    for _ in range(3):
        rows.append(("v0 host post", timed(v0, frames)))
        rows.append(("v1 GPU post", timed(vgpu(m1), frames)))
        rows.append(("v2 fused + tiled", timed(vgpu(m2), frames)))
    agg = {}
    for name, ms in rows:
        agg.setdefault(name, []).append(ms)

    print(f"\n{w}x{h}, {n - 8} timed frames, median of interleaved repeats")
    print(f"  {'version':20s} {'ms/frame':>9s} {'FPS':>7s} {'vs v0':>8s}")
    base = np.median(agg["v0 host post"])
    for name in ("v0 host post", "v1 GPU post", "v2 fused + tiled"):
        ms = np.median(agg[name])
        print(f"  {name:20s} {ms:9.2f} {1000 / ms:7.1f} {base / ms:7.2f}x")

    # equality is not optional: v1/v2 are speedups, not different algorithms
    a = v0(frames[-1])
    b = vgpu(m1)(frames[-1])
    c = vgpu(m2)(frames[-1])
    print(f"  masks identical: v0==v1 {np.array_equal(a, b)}, "
          f"v1==v2 {np.array_equal(b, c)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="+", type=int, default=[480, 720, 1080])
    args = ap.parse_args()
    if GMM_Mask_CUDA is None:
        raise SystemExit("no CUDA device — this benchmark needs a real GPU")
    for s in args.sizes:
        bench(SIZES[s])
