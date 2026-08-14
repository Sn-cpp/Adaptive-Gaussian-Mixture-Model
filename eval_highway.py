"""Score mask pipelines against CDnet `highway` ground truth.

This is the harness that decides what ships. Nothing about mask quality should
be argued from a screenshot — run this and read the F1 column.

    HIGHWAY_DIR=/path/to/highway python eval_highway.py

The directory must hold CDnet's own layout::

    highway/input/in000001.jpg ...      groundtruth/gt000001.png ...
    highway/ROI.bmp                     temporalROI.txt

Scoring follows the CDnet protocol exactly: only frames inside `temporalROI`
(470-1700 for highway), and only pixels whose ground truth is 0 or 255 inside
`ROI.bmp` — CDnet marks shadows (50) and unknown boundaries (170) as
don't-care, and counting them is the single easiest way to publish a wrong
number.
"""
import argparse
import os
import time

import cv2
import numpy as np

from gmm_mask import GMM_Mask_Numba
from settings import MOG2_BG_PROB_THRESHOLD

DEFAULT_DIR = os.environ.get("HIGHWAY_DIR", "highway")
T0, T1 = 470, 1700


# ── the chains under test ─────────────────────────────────────────────────────

def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask):
    """Fill every background region fully enclosed by foreground.

    Flood the background inward from a one-pixel border added around the frame,
    then OR back the complement — whatever the flood could not reach is a hole.
    The border matters: seeding at (0, 0) breaks the moment an object touches
    that corner, because then the flood cannot start and the complement is the
    whole image.
    """
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(padded)[1:h + 1, 1:w + 1]
    return mask | holes


def sobel_edges(frame_f32):
    """Tin's edge gate: |dI/dx| and |dI/dy| averaged, as uint8."""
    gray = cv2.cvtColor(frame_f32, cv2.COLOR_BGR2GRAY)
    gx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    gy = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    return cv2.addWeighted(gx, 0.5, gy, 0.5, 0)


def connect_foreground(mask, k_close, k_dilate, k_erode):
    """Tin's contour fill: grow, close, fill every outline solid, shrink back."""
    closed = cv2.morphologyEx(cv2.dilate(mask, k_dilate), cv2.MORPH_CLOSE, k_close)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    filled = np.zeros_like(mask)
    cv2.fillPoly(filled, contours, 255)
    return cv2.erode(filled, k_erode)


def connect_foreground_v2(mask, k_close, k_dilate, k_erode, seed=(50, 50)):
    """Tin's `1fe1c5e` version: contour fill, then keep only the largest blob
    and fill its holes, then shrink again.

    The hole fill is `bitwise_not` -> `floodFill` from a seed -> `bitwise_or`.
    Seed is a fixed pixel in his code; kept as a parameter here so the failure
    mode is measurable rather than argued about — if the seed lands *inside*
    the object it is already black, filling black with black does nothing, the
    outer background survives the OR, and the mask explodes to almost the whole
    frame.
    """
    result = connect_foreground(mask, k_close, k_dilate, k_erode)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    if n <= 1:
        return result
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    lr = np.where(labels == largest, np.uint8(255), np.uint8(0))

    inv = cv2.bitwise_not(lr)
    h, w = inv.shape[:2]
    cv2.floodFill(inv, np.zeros((h + 2, w + 2), np.uint8), seed, 0,
                  (10,) * 3, (10,) * 3, flags=4 | cv2.FLOODFILL_FIXED_RANGE)
    return cv2.erode(cv2.bitwise_or(lr, inv), k_erode)


def build_chains(h, w):
    """name -> fn(mask, bg_prob, sobel) -> uint8 mask in {0, 255}."""
    short = min(h, w)
    k_close = _ellipse(2 * max(3, int(short * 0.019)) + 1)
    k_dilate = _ellipse(2 * max(5, int(short * 0.042)) + 1)
    k_erode = _ellipse(2 * max(3, int(short * 0.027)) + 1)
    k15 = _ellipse(15)

    def binary(m):
        return np.where(m == 255, np.uint8(255), np.uint8(0))

    def gate(bp, sob, t):
        return np.where((sob > 0) & (bp <= t), np.uint8(255), np.uint8(0))

    def tin_gate(bp, sob):
        """`1fe1c5e` exactly: an edge of real strength, and low confidence."""
        return np.where((sob >= 10) & (bp <= 0.2), np.uint8(255), np.uint8(0))

    def prob(bp, t=MOG2_BG_PROB_THRESHOLD):
        return np.where(bp < t, np.uint8(255), np.uint8(0))

    return {
        "raw mask":
            lambda m, bp, s: binary(m),
        "mask + median5":
            lambda m, bp, s: cv2.medianBlur(binary(m), 5),
        "mask + median5 + fill":
            lambda m, bp, s: fill_holes(cv2.medianBlur(binary(m), 5)),
        "bg_prob":
            lambda m, bp, s: prob(bp),
        "bg_prob + median5":
            lambda m, bp, s: cv2.medianBlur(prob(bp), 5),
        "bg_prob + median5 + fill":
            lambda m, bp, s: fill_holes(cv2.medianBlur(prob(bp), 5)),
        "bg_prob + median5 + close15 + fill":
            lambda m, bp, s: fill_holes(cv2.morphologyEx(
                cv2.medianBlur(prob(bp), 5), cv2.MORPH_CLOSE, k15, iterations=1)),
        "sobel gate 0.4":
            lambda m, bp, s: gate(bp, s, 0.4),
        "sobel gate + median3":
            lambda m, bp, s: cv2.medianBlur(gate(bp, s, 0.4), 3),
        "sobel gate + median3 + fill":
            lambda m, bp, s: fill_holes(cv2.medianBlur(gate(bp, s, 0.4), 3)),
        "Tin dd40d92: gate.4 + med3 + contour":
            lambda m, bp, s: connect_foreground(
                cv2.medianBlur(gate(bp, s, 0.4), 3), k_close, k_dilate, k_erode),
        # 1fe1c5e tightened both gate thresholds and added largest-blob +
        # hole fill. `tin_gate` is his exact rule: sobel >= 10 AND bg_prob <= 0.2.
        "Tin 1fe1c5e: gate.2 + med5 + contour+fill":
            lambda m, bp, s: connect_foreground_v2(
                cv2.medianBlur(tin_gate(bp, s), 5), k_close, k_dilate, k_erode),
        "Tin 1fe1c5e, no largest-blob step":
            lambda m, bp, s: connect_foreground(
                cv2.medianBlur(tin_gate(bp, s), 5), k_close, k_dilate, k_erode),
    }


# ── scoring ───────────────────────────────────────────────────────────────────

def score(root, colorspace="bgr", t0=T0, t1=T1, limit_chains=None):
    roi = cv2.imread(os.path.join(root, "ROI.bmp"), 0) > 0
    first = cv2.imread(os.path.join(root, "input", "in000001.jpg"))
    if first is None:
        raise SystemExit(f"no frames under {root}/input — set HIGHWAY_DIR")
    h, w = first.shape[:2]

    cvt = ((lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb))
           if colorspace == "ycrcb" else (lambda f: f))
    model = GMM_Mask_Numba(h, w)

    chains = build_chains(h, w)
    if limit_chains:
        chains = {k: v for k, v in chains.items() if k in limit_chains}
    acc = {k: np.zeros(3) for k in chains}
    empty = dict.fromkeys(chains, 0)
    secs = dict.fromkeys(chains, 0.0)
    scored = 0

    for i in range(1, t1 + 1):
        bgr = cv2.imread(os.path.join(root, "input", f"in{i:06d}.jpg"))
        frame = np.ascontiguousarray(cvt(bgr), dtype=np.float32)
        mask, bg_prob, _ = model.apply(frame)
        if i < t0:
            continue
        # the Sobel gate reads image texture, so it always sees BGR
        sob = sobel_edges(np.ascontiguousarray(bgr, dtype=np.float32))
        gt = cv2.imread(os.path.join(root, "groundtruth", f"gt{i:06d}.png"), 0)
        valid = roi & ((gt == 255) | (gt == 0))
        g = (gt == 255) & valid

        for name, fn in chains.items():
            t = time.perf_counter()
            out = fn(np.asarray(mask), np.asarray(bg_prob), sob)
            secs[name] += time.perf_counter() - t
            p = (out == 255) & valid
            acc[name] += [np.sum(p & g), np.sum(p & ~g), np.sum(~p & g)]
            empty[name] += not (out == 255).any()
        scored += 1

    print(f"\nhighway {t0}-{t1} ({scored} frames), model input = {colorspace}")
    print(f"  {'chain':36s} {'F1':>7s} {'IoU':>7s} {'P':>7s} {'R':>7s} "
          f"{'empty':>6s} {'ms/f':>7s}")
    rows = []
    for name, (tp, fp, fn_) in acc.items():
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn_, 1)
        rows.append((2 * p * r / max(p + r, 1e-9), tp / max(tp + fp + fn_, 1),
                     p, r, empty[name], secs[name] / max(scored, 1) * 1000, name))
    for f1, iou, p, r, e, ms, name in sorted(rows, reverse=True):
        print(f"  {name:36s} {f1:7.4f} {iou:7.4f} {p:7.4f} {r:7.4f} "
              f"{e:6d} {ms:7.2f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--colorspace", default="both", choices=("bgr", "ycrcb", "both"))
    ap.add_argument("--last-frame", type=int, default=T1)
    args = ap.parse_args()

    spaces = ("bgr", "ycrcb") if args.colorspace == "both" else (args.colorspace,)
    for cs in spaces:
        score(args.dir, cs, t1=args.last_frame)
