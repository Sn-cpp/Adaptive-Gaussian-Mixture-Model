"""
bench_cuda_v0.py
Performance benchmark for GrabCut_CUDA_v0 (CUDA push-relabel) on footage.mp4.

Measures per-step wall-clock time across N_FRAMES frames, then prints:
  - per-step mean / std / min / max
  - total pipeline mean (ms)
  - comparison table against the baseline timings supplied in the docstring

Run:
    python bench_cuda_v0.py [--input footage.mp4] [--frames 50] [--warmup 5]
"""

import argparse
import time
import numpy as np
import cv2

# ── JIT warm-up (must happen before any real measurement) ─────────────────────
from gmm_em import warmup_em_gmm_jit
from gmm_mask import warmup_mask_gmm_jit
warmup_em_gmm_jit()
warmup_mask_gmm_jit()

# Import CUDA grabcut and trigger its own warmup
from grabcut.gpu.cuda_v0.grabcut_cuda_v0 import (
    GrabCut_CUDA_v0,
    make_gc_mask, calc_beta, calc_nweights,
    build_tlinks, build_nlinks, compose_blur,
    warmup_grabcut_jit,
)
from grabcut.gpu.cuda_v0.push_relabel_cuda_v0 import push_relabel, warmup_push_relabel
from grabcut.gpu.cuda_v0.morphology_cuda_v0 import (
    morphological_close, morphological_open, largest_component, warmup_morph,
)
warmup_push_relabel()
warmup_morph()
warmup_grabcut_jit()

from gmm_em import GMM_EM_Numba_CPU
from gmm_mask import GMM_Mask_Numba
from settings import PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ, LAM_FACTOR

# ── Baseline numbers (ms) from the original CPU-parallel implementation ────────
BASELINE = {
    "make_gc":        0.398,
    "bg fit":         3.447,
    "fg fit":         3.711,
    "bg nlp":         1.754,
    "fg nlp":         1.650,
    "calc beta":     13.988,
    "calc nweights": 16.772,
    "max nw":         0.121,
    "build tlinks":   0.203,
    "build nlinks":   0.311,
    "push relabel":  168.768,
    "morph close":    3.498,
    "morph open":     2.392,
    "large component": 1.808,
}

STEPS = list(BASELINE.keys())


def _ms() -> float:
    return time.perf_counter() * 1e3


def benchmark_frame(frame, bg_prob, gc):
    """Run one frame through GrabCut_CUDA_v0 with per-step timing.

    Mirrors the body of apply() exactly so timings are comparable.
    Returns dict[step_name -> elapsed_ms].
    """
    H, W = gc.H, gc.W
    t = {}

    t0 = _ms()
    make_gc_mask(bg_prob, gc._gc_mask)
    t["make_gc"] = _ms() - t0

    img_f32 = frame.astype(np.float32)

    t0 = _ms()
    gc._bg_gmm.fit(img_f32, gc._gc_mask, is_fg=False)
    t["bg fit"] = _ms() - t0

    t0 = _ms()
    gc._fg_gmm.fit(img_f32, gc._gc_mask, is_fg=True)
    t["fg fit"] = _ms() - t0

    t0 = _ms()
    gc._bg_gmm.neg_log_prob(img_f32, gc._nlp_bg)
    t["bg nlp"] = _ms() - t0

    t0 = _ms()
    gc._fg_gmm.neg_log_prob(img_f32, gc._nlp_fg)
    t["fg nlp"] = _ms() - t0

    t0 = _ms()
    beta = calc_beta(img_f32)
    t["calc beta"] = _ms() - t0

    t0 = _ms()
    calc_nweights(img_f32, beta, gc.gamma,
                  gc._leftW, gc._upleftW, gc._upW, gc._uprightW)
    t["calc nweights"] = _ms() - t0

    t0 = _ms()
    max_nw = max(float(gc._leftW.max()), float(gc._upW.max()))
    lam = (max_nw * LAM_FACTOR if max_nw > 0.0 else float(gc.gamma) * float(LAM_FACTOR))
    t["max nw"] = _ms() - t0

    t0 = _ms()
    build_tlinks(gc._gc_mask, gc._nlp_bg, gc._nlp_fg, lam,
                 gc._cap_src, gc._cap_snk)
    t["build tlinks"] = _ms() - t0

    t0 = _ms()
    build_nlinks(gc._leftW.astype(np.float32),
                 gc._upW.astype(np.float32),
                 np.int32(H), np.int32(W),
                 gc._cap_right, gc._cap_down)
    t["build nlinks"] = _ms() - t0

    t0 = _ms()
    labeling = push_relabel(
        gc._cap_src, gc._cap_snk,
        gc._cap_right, gc._cap_down,
        np.int32(H), np.int32(W),
        PUSH_RELABEL_MAX_ITER, PUSH_RELABEL_RELABEL_FREQ,
    )
    t["push relabel"] = _ms() - t0

    fg = (labeling == 0).reshape(H, W).astype(np.uint8)
    np.multiply(fg, np.uint8(255), out=gc._final_mask)

    t0 = _ms()
    morphological_close(gc._final_mask, gc._morph_tmp1, gc._morph_tmp2,
                        np.int32(H), np.int32(W), radius=3)
    t["morph close"] = _ms() - t0

    t0 = _ms()
    morphological_open(gc._morph_tmp2, gc._morph_tmp1, gc._final_mask,
                       np.int32(H), np.int32(W), radius=2)
    t["morph open"] = _ms() - t0

    t0 = _ms()
    np.copyto(gc._final_mask, largest_component(gc._final_mask,
                                                 np.int32(H), np.int32(W)))
    t["large component"] = _ms() - t0

    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="footage.mp4")
    parser.add_argument("--frames", type=int, default=50,
                        help="Number of frames to benchmark (after warmup)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Frames to discard before measuring")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video : {args.input}  {W}x{H}  {fps:.1f} fps  {total_video_frames} frames")
    print(f"Bench : {args.warmup} warmup + {args.frames} measured frames\n")

    gmm_mask = GMM_Mask_Numba(H, W)
    gc = GrabCut_CUDA_v0(H, W, GMM_EM_Numba_CPU(H, W), GMM_EM_Numba_CPU(H, W))

    records = {s: [] for s in STEPS}

    frame_idx = 0
    measured  = 0

    while measured < args.frames:
        ret, frame = cap.read()
        if not ret:
            # Loop the video if it's shorter than requested frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        _, bg_prob, _ = gmm_mask.apply(frame, to_host=True)

        timing = benchmark_frame(frame, bg_prob, gc)

        frame_idx += 1
        if frame_idx <= args.warmup:
            continue   # discard warmup frames

        for s in STEPS:
            records[s].append(timing[s])
        measured += 1
        print(f"\rMeasured {measured}/{args.frames}", end="", flush=True)

    cap.release()
    print("\n")

    # ── Results ───────────────────────────────────────────────────────────────
    col_w = 18
    print(f"{'Step':<18} {'Baseline':>10} {'CUDA mean':>10} {'std':>8} {'min':>8} {'max':>8} {'speedup':>9}")
    print("-" * 73)

    total_base = sum(BASELINE.values())
    total_cuda_means = 0.0

    for s in STEPS:
        arr  = np.array(records[s])
        mean = arr.mean()
        std  = arr.std()
        mn   = arr.min()
        mx   = arr.max()
        base = BASELINE[s]
        spd  = base / mean if mean > 0 else float("inf")
        total_cuda_means += mean
        print(f"{s:<18} {base:>10.3f} {mean:>10.3f} {std:>8.3f} {mn:>8.3f} {mx:>8.3f} {spd:>8.2f}x")

    print("-" * 73)
    total_spd = total_base / total_cuda_means if total_cuda_means > 0 else float("inf")
    print(f"{'TOTAL':<18} {total_base:>10.3f} {total_cuda_means:>10.3f} {'':>8} {'':>8} {'':>8} {total_spd:>8.2f}x")
    print(f"\n(all times in ms, speedup = baseline / CUDA mean)\n")


if __name__ == "__main__":
    main()
