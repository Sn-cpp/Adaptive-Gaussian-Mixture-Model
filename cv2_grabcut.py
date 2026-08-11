"""Side-by-side comparison: cv2.grabCut vs our MOG2+dual-GMM+Push-Relabel pipeline.

Mirrors cv2_grabcut2.py behaviour:
  ROI      : 60% × 70% centred rectangle (same as reference script)
  Output   : sharp foreground composited over Gaussian-blurred background
  Interval : save PNG every 5 frames

Saved panels per frame (hstack):
  [Original + ROI] | [cv2 composite] | [cv2 mask] | [Our composite] | [Our mask]

Usage:
  conda run -n AutoEncoder python cv2_grabcut.py [video] [scale]

  video : path to video file (default: footage.mp4)
  scale : resize factor, e.g. 0.25 for quarter-res (default: 0.25)
"""
import os
import sys
import cv2
import numpy as np

VIDEO_PATH  = sys.argv[1] if len(sys.argv) > 1 else "footage.mp4"
SCALE       = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

SAVE_EVERY  = 5          # save a comparison PNG every N frames
ITER_COUNT  = 2          # cv2.grabCut iterations (same as cv2_grabcut2.py)
WARMUP      = 10         # MOG2 warm-up frames before Push-Relabel starts
MAX_FRAMES  = 300        # cap run length (None = full video)
BLUR_KSIZE  = 15         # Gaussian blur kernel size for background
OUT_DIR     = "grabcut_ref"

os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from gmm import GMM_CPU_NUMBA
from gmm.mog2_common import to_planar
from grabcut_numba import GrabCutPipeline
from settings import MOG2_N_COMPONENTS

# ── Open video ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

src_W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_s  = cap.get(cv2.CAP_PROP_FPS)
print(f"Source: {src_W}x{src_H}  {fps_s:.1f}fps  {total} frames")

W = max(1, int(round(src_W * SCALE)))
H = max(1, int(round(src_H * SCALE)))
print(f"Working: {W}x{H}  (scale={SCALE})")


def resize_frame(f):
    if SCALE == 1.0:
        return f
    return cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)


# ── ROI: 60% × 70% centred (matches cv2_grabcut2.py) ────────────────────────
rw = int(W * 0.6); rh = int(H * 0.7)
rx = (W - rw) // 2; ry = (H - rh) // 2
rect = (rx, ry, rw, rh)
print(f"ROI: {rect}   warmup={WARMUP}  max_frames={MAX_FRAMES}")

# ── Init our pipeline ────────────────────────────────────────────────────────
ret, first_raw = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")
first = resize_frame(first_raw)

print("Initialising GMM + Push-Relabel (Numba JIT warmup ~30 s)...")
gmm      = GMM_CPU_NUMBA(first, MOG2_N_COMPONENTS)
pipeline = GrabCutPipeline(gmm, rect, blur_ksize=BLUR_KSIZE)
print("JIT warmup done.")

# ── cv2.grabCut buffers (reused per frame) ───────────────────────────────────
gc_mask   = np.zeros((H, W), dtype=np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)


def cv2_grabcut_composite(frame):
    """Run cv2.grabCut and return (fg_mask_255, composite_blurred_bg)."""
    gc_mask[:] = 0; bgd_model[:] = 0; fgd_model[:] = 0
    cv2.grabCut(frame, gc_mask, rect, bgd_model, fgd_model,
                ITER_COUNT, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                     np.uint8(1), np.uint8(0))
    bg = cv2.GaussianBlur(frame, (BLUR_KSIZE, BLUR_KSIZE), 0)
    cv2.copyTo(frame, mask2[:, :, np.newaxis].astype(bool), bg)
    fg_u8 = mask2 * np.uint8(255)
    return fg_u8, bg


# ── Main loop ────────────────────────────────────────────────────────────────
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0

while True:
    ret, raw = cap.read()
    if not ret:
        break
    frame_idx += 1
    if MAX_FRAMES and frame_idx > MAX_FRAMES:
        break

    frame = resize_frame(raw)

    # cv2.grabCut (runs every frame, no warmup needed)
    cv2_fg, cv2_composite = cv2_grabcut_composite(frame.copy())

    # Our pipeline
    if frame_idx <= WARMUP:
        gmm.step(to_planar(frame))
        our_mask      = np.zeros((H, W), dtype=np.uint8)
        our_composite = frame.copy()
        status = f"warmup {frame_idx}/{WARMUP}"
    else:
        mog2_mask, bg_prob, our_mask, our_composite, t = pipeline.rqstep(frame)
        status = f"frame {frame_idx}  ({t*1000:.0f}ms)"

    # Save every SAVE_EVERY frames (and right after warmup ends)
    if frame_idx % SAVE_EVERY == 0 or frame_idx == WARMUP + 1:
        p_orig = frame.copy()
        cv2.rectangle(p_orig, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.putText(p_orig, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        p_cv2_comp = cv2_composite.copy()
        cv2.putText(p_cv2_comp, "cv2 composite", (10, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        p_cv2_mask = cv2.cvtColor(cv2_fg, cv2.COLOR_GRAY2BGR)
        cv2.putText(p_cv2_mask, "cv2 mask", (10, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        p_our_comp = our_composite.copy()
        cv2.putText(p_our_comp, "ours composite", (10, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        p_our_mask = cv2.cvtColor(our_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(p_our_mask, "ours mask", (10, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        out = np.hstack([p_orig, p_cv2_comp, p_cv2_mask, p_our_comp, p_our_mask])
        path = os.path.join(OUT_DIR, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(path, out)

        cv2_px = int(cv2_fg.sum() // 255)
        our_px = int(our_mask.sum() // 255)
        print(f"  [{status:>25}]  cv2_fg={cv2_px:6d}  ours_fg={our_px:6d}  → {path}")

print(f"\nDone. {frame_idx} frames. Output in ./{OUT_DIR}/")
cap.release()
