"""cpu_baseline.py — the course-template entry point for the CPU reference.

The grading rubric asks for exactly this file, runnable as::

    python src/cpu_baseline.py

on a machine with no GPU. It drives the pure-NumPy sequential model
(`gmm_mask/cpu/gmm_mask_cpu.py`) plus the host post-processing chain over a
synthetic traffic sequence, reports wall-clock timing in the template's
format, and verifies the mask against OpenCV's own MOG2 — the "trusted
library" reference the template calls for.

Two honest notes, so this file does not claim more than it does:

* The algorithm itself is pure Python + NumPy. Importing it through the
  package does pull in numba (the package also ships the Numba baseline, and
  its availability probe calls `numba.cuda.is_available()`, which is harmless
  without a driver). The guarantee that matters is behavioural: this script
  runs to completion on a machine with no GPU, which is how it was verified.
* The synthetic sequence exists so the baseline runs anywhere with zero
  downloads. The scored quality numbers come from CDnet via
  `eval_highway.py`; this file is the timing-and-correctness entry point,
  not the quality benchmark.

The full pipeline lives behind `main.py --model cpu`; the GPU versions climb
from this exact function.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np


def load_data(n_frames: int = 12, size=(320, 240)):
    """Synthetic traffic frames + the cv2.MOG2 reference masks for them."""
    import bench_post
    from settings import (MOG2_BACKGROUND_RATIO, MOG2_HISTORY,
                          MOG2_N_COMPONENTS, MOG2_VAR_THRESHOLD)

    frames = bench_post.make_frames(n_frames, size)
    ref = cv2.createBackgroundSubtractorMOG2(
        history=int(MOG2_HISTORY), varThreshold=float(MOG2_VAR_THRESHOLD),
        detectShadows=False)
    ref.setNMixtures(int(MOG2_N_COMPONENTS))
    ref.setBackgroundRatio(float(MOG2_BACKGROUND_RATIO))
    truth = [ref.apply(cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)) for f in frames]
    return frames, truth


def run_cpu(frames):
    """The sequential pipeline: model + threshold + median + fill, all host."""
    from gmm_mask import GMM_Mask_CPU
    from utils.post_processing import refine_mask

    h, w = frames[0].shape[:2]
    model = GMM_Mask_CPU(h, w)
    raw, refined = [], []
    for f in frames:
        mask, bg_prob, _ = model.apply(cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb))
        raw.append(np.asarray(mask).copy())
        refined.append(refine_mask(np.asarray(mask), bg_prob=np.asarray(bg_prob)))
    return raw, refined


def verify(raw_masks, truth) -> float:
    """Fraction of pixels agreeing with cv2's MOG2 across every frame."""
    agree = sum(int((a == b).sum()) for a, b in zip(raw_masks, truth))
    total = sum(b.size for b in truth)
    return agree / total


if __name__ == "__main__":
    frames, truth = load_data()
    arr = np.stack(frames)
    print(f"Input shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Input size: {arr.nbytes / 1e6:.1f} MB")

    t0 = time.perf_counter()
    raw, refined = run_cpu(frames)
    elapsed = time.perf_counter() - t0
    accuracy = verify(raw, truth)

    print("\nCPU baseline results:")
    print(f"  Time:       {elapsed:.3f} s  ({elapsed / len(frames) * 1000:.1f} ms/frame)")
    print(f"  Agreement with cv2.MOG2: {accuracy:.6f}")
    print(f"  Throughput: {len(frames) / elapsed:.2f} frames/s")
    assert accuracy > 0.999, "sequential model no longer matches OpenCV"
    print("\nOK — this is the reference every GPU version is tested against.")
