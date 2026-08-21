"""Throughput on real Full-HD traffic footage, across resolutions.

    python benchmarks/bench_realfootage.py                     # all versions
    python benchmarks/bench_realfootage.py --versions numba    # CPU-only smoke

Every other timing table in this project runs on synthetic frames. This one
answers the question a referee should ask about them: does the speedup story
survive a real decoder and real footage? One fixed-camera 1080p traffic clip
(Pexels 4791721, free license, 2505 frames at 29.97 FPS) is decoded once and
downscaled to each tier — so across the resolution axis the *scene is
identical* and only the pixel count changes, which is the only honest way to
present a resolution-scaling demo.

The clip is fetched on first run and cached beside the repo (it is in
.gitignore; the URL and its verified properties are pinned here instead).
Quality is *not* scored on this clip — it has no ground truth. Quality lives
on CDnet highway (`benchmarks/records/eval_highway_full.txt`); this file is
throughput only.
"""
import argparse
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np

CLIP_URL = ("https://videos.pexels.com/video-files/4791721/"
            "4791721-hd_1920_1080_30fps.mp4")
CLIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "traffic1080.mp4")
# verified 2026-08-20: 1920x1080, 2505 frames, 29.97 FPS, 57.2 MB
TIERS = [("240p", (320, 240)), ("480p", (854, 480)),
         ("720p", (1280, 720)), ("1080p", (1920, 1080))]


def fetch_clip():
    if not os.path.exists(CLIP_PATH):
        print(f"fetching {CLIP_URL} ...", flush=True)
        # Pexels rejects urllib's default User-Agent with 403
        req = urllib.request.Request(CLIP_URL,
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(CLIP_PATH, "wb") as f:
            import shutil
            shutil.copyfileobj(r, f)
    cap = cv2.VideoCapture(CLIP_PATH)
    w, h = int(cap.get(3)), int(cap.get(4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert (w, h) == (1920, 1080), f"unexpected clip geometry {w}x{h}"
    print(f"clip: {w}x{h}, {n} frames")


def decode(size, n_frames, skip=60):
    """n_frames at `size`, skipping the first `skip` (encoder warm-up glare)."""
    cap = cv2.VideoCapture(CLIP_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
    out = []
    while len(out) < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        out.append(np.ascontiguousarray(cv2.resize(f, size)))
    cap.release()
    return out


def one_pass(build, fn, frames, warm=8, sync=False):
    """bench_post.one_pass, minus the unconditional cuda.synchronize --
    so the CPU-only smoke run works on a machine with no CUDA driver."""
    def _sync():
        if sync:
            from numba import cuda
            cuda.synchronize()
    m = build()
    for f in frames[:warm]:
        fn(m, f)
    _sync()
    t0 = time.perf_counter()
    for f in frames[warm:]:
        fn(m, f)
    _sync()
    return (time.perf_counter() - t0) / len(frames[warm:]) * 1000


def main():
    import bench_post as bp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--versions", nargs="+",
                    default=["numba", "v0", "v1", "v2"],
                    choices=["numba", "v0", "v1", "v2"])
    args = ap.parse_args()

    from gmm_mask import (GMM_Mask_CUDA, GMM_Mask_CUDA_v1, GMM_Mask_CUDA_v2,
                          GMM_Mask_Numba)
    builders = {"numba": (GMM_Mask_Numba, bp.run_v0),
                "v0": (GMM_Mask_CUDA, bp.run_v0),
                "v1": (GMM_Mask_CUDA_v1, bp.run_gpu),
                "v2": (GMM_Mask_CUDA_v2, bp.run_gpu)}
    gpu_wanted = any(v != "numba" for v in args.versions)
    if gpu_wanted and GMM_Mask_CUDA is None:
        raise SystemExit("GPU versions requested but no CUDA device")

    fetch_clip()
    bp.environment()

    print(f"\nreal footage (Pexels 4791721), {args.frames - 8} timed frames "
          f"per cell, decode excluded from timing")
    print(f"  {'tier':>6} " + "".join(f"{v:>14}" for v in args.versions))
    for tier, size in TIERS:
        frames = decode(size, args.frames)
        if len(frames) < args.frames:
            print(f"  {tier:>6}  (short decode: {len(frames)} frames)")
        row = f"  {tier:>6} "
        for v in args.versions:
            cls, fn = builders[v]
            h, w = frames[0].shape[:2]
            ms = one_pass(lambda c=cls: c(h, w), fn, frames,
                          sync=(v != "numba"))
            row += f"{ms:8.2f}ms" + f"{1000/ms:5.0f} "
        print(row)
    print("\n  (columns: ms/frame then FPS. Same scene at every tier — only the")
    print("   pixel count changes. Quality is scored on CDnet, not on this clip.)")


if __name__ == "__main__":
    main()
