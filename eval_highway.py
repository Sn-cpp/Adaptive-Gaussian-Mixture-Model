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

from gmm_mask import (GMM_Mask_CPU, GMM_Mask_CUDA, GMM_Mask_CUDA_v1,
                      GMM_Mask_CUDA_v2, GMM_Mask_CuPy, GMM_Mask_Numba)
from settings import MOG2_BG_PROB_THRESHOLD
from settings import BLUR_KSIZE, BLUR_SIGMA
from utils.post_processing import background_blur, fill_holes, refine_mask

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


MODELS = {
    "cpu": GMM_Mask_CPU,
    "numba": GMM_Mask_Numba,
    "cuda": GMM_Mask_CUDA,
    "cuda_v1": GMM_Mask_CUDA_v1,
    "cuda_v2": GMM_Mask_CUDA_v2,
    "cupy": GMM_Mask_CuPy,
}


def build_model(name, h, w, post=False, colorspace="ycrcb"):
    cls = MODELS[name]
    if cls is None:
        raise SystemExit(f"--model {name} is unavailable on this machine")
    # v1/v2 default to running the post chain on the device. For `score()` the
    # host chains under test must all see the same raw model output, so the
    # device post-processing is switched off there. Signature inspection, not
    # try/except TypeError: that would also catch a real TypeError from inside
    # __init__ and silently leave GPU post-processing on.
    import inspect
    params = inspect.signature(cls.__init__).parameters
    kw = {}
    if "post" in params:
        kw["post"] = post
    if "colorspace" in params:
        kw["colorspace"] = colorspace
    return cls(h, w, **kw)


def parity(root, model_name, ref_name="numba", colorspace="ycrcb",
           t0=T0, t1=T1):
    """Per-frame `np.array_equal` between the shipping mask of two backends.

    This exists because the obvious check is not one, and because the first
    version of this function was not one either.

    *Not* an unchanged F1: F1 is a four-decimal summary of two million pixels,
    and a mask can move by hundreds of them without shifting it. And `score()`
    used to hardcode the Numba model, so running it after moving work onto the
    GPU exercised the CPU path and cheerfully reported that the CPU path still
    worked.

    Nor is it enough to compare `apply()` with the device post-processing
    switched off. That was this function's first mistake: it compared the raw
    MOG2 decision, while the mask the pipeline actually ships comes from
    `bg_prob < threshold` followed by a median, and — for v1/v2 — from a colour
    conversion that now happens on the device. A backend could convert colour
    wrongly, or tile the median wrongly, and still pass.

    So this drives each backend through the path `main.py` drives it through:
    `mask_from_bgr` -> device post chain -> host `fill_holes` where the backend
    supports it, and host `cvtColor` -> `apply` -> `refine_mask` where it does
    not. Then it compares the mask that would have been composited.
    """
    first = cv2.imread(os.path.join(root, "input", "in000001.jpg"))
    if first is None:
        raise SystemExit(f"no frames under {root}/input — set HIGHWAY_DIR")
    h, w = first.shape[:2]

    def shipping_chain(name):
        """Return `fn(bgr) -> (refined mask, composite)` via the backend's real path.

        The composite matters as much as the mask. A backend could produce an
        identical mask and still blur differently — that is precisely what
        Kernel 2 changed — so comparing only the mask would leave the half of
        the pipeline this project actually rewrote unchecked.
        """
        m = build_model(name, h, w, post=True, colorspace=colorspace)
        if hasattr(m, "mask_from_bgr"):
            def gpu(bgr):
                refined = fill_holes(m.mask_from_bgr(bgr))
                return refined, m.composite(refined)
            return m, gpu

        cvt = ((lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb))
               if colorspace == "ycrcb" else (lambda f: f))

        def host(bgr):
            mask, bg_prob, _ = m.apply(cvt(bgr))
            refined = refine_mask(np.asarray(mask), bg_prob=np.asarray(bg_prob))
            return refined, background_blur(bgr, refined, BLUR_KSIZE, BLUR_SIGMA)
        return m, host

    _, run_a = shipping_chain(model_name)
    _, run_b = shipping_chain(ref_name)

    bad = bad_c = worst = scored = 0
    first_bad = None
    saw_fg = False
    for i in range(1, t1 + 1):
        bgr = cv2.imread(os.path.join(root, "input", f"in{i:06d}.jpg"))
        ma, ca = run_a(bgr)
        mb, cb = run_b(bgr)
        if i < t0:
            continue
        saw_fg |= bool((mb == 255).any())
        n = int((np.asarray(ma) != np.asarray(mb)).sum())
        nc = int((np.asarray(ca) != np.asarray(cb)).sum())
        if (n or nc) and first_bad is None:
            first_bad = i
        bad += n
        bad_c += nc
        worst = max(worst, n)
        scored += 1

    px = scored * h * w
    print(f"\nparity {model_name} vs {ref_name}, {colorspace}, shipping mask and "
          f"composite, frames {t0}-{t1} ({scored} frames, {px} pixels each)")
    print(f"  mask differing pixels   : {bad}  ({bad / max(px, 1):.6%})")
    print(f"  composite differing px  : {bad_c}  ({bad_c / max(px * 3, 1):.6%})")
    print(f"  worst mask frame       : {worst} px")
    print(f"  first difference       : {first_bad if first_bad else 'none'}")
    if not saw_fg:
        print("  DEGENERATE             : no foreground in any frame — proves nothing")
        return 1
    print(f"  VERDICT                : "
          f"{'IDENTICAL' if bad == 0 and bad_c == 0 else 'DIVERGED'}")
    return bad + bad_c


def scorable(gt, roi):
    """The pixels CDnet says may be scored at all.

    CDnet labels shadows 50 and object boundaries 170 and defines both as
    *don't care*, and it ships an ROI mask. Counting either is the easiest way
    in this project to publish a confident, reproducible, wrong number, so the
    rule lives in one function that `tests/test_scoring.py` imports and checks
    against a fixture whose answer is countable by hand.
    """
    return roi & ((gt == 255) | (gt == 0))


def confusion(pred, gt, roi):
    """(TP, FP, FN) over the scorable pixels only."""
    valid = scorable(gt, roi)
    g = (gt == 255) & valid
    p = (pred == 255) & valid
    return int(np.sum(p & g)), int(np.sum(p & ~g)), int(np.sum(~p & g))


def metrics(tp, fp, fn):
    """(F1, IoU, precision, recall). Zero-safe: an empty mask scores 0."""
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return (2 * p * r / max(p + r, 1e-9), tp / max(tp + fp + fn, 1), p, r)


# ── scoring ───────────────────────────────────────────────────────────────────

def score(root, colorspace="bgr", t0=T0, t1=T1, limit_chains=None,
          model_name="numba"):
    roi = cv2.imread(os.path.join(root, "ROI.bmp"), 0) > 0
    first = cv2.imread(os.path.join(root, "input", "in000001.jpg"))
    if first is None:
        raise SystemExit(f"no frames under {root}/input — set HIGHWAY_DIR")
    h, w = first.shape[:2]

    cvt = ((lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb))
           if colorspace == "ycrcb" else (lambda f: f))
    model = build_model(model_name, h, w)

    chains = build_chains(h, w)
    if limit_chains:
        chains = {k: v for k, v in chains.items() if k in limit_chains}
    acc = {k: np.zeros(3) for k in chains}
    empty = dict.fromkeys(chains, 0)
    secs = dict.fromkeys(chains, 0.0)
    scored = 0

    for i in range(1, t1 + 1):
        bgr = cv2.imread(os.path.join(root, "input", f"in{i:06d}.jpg"))
        frame = cvt(bgr)
        mask, bg_prob, _ = model.apply(frame)
        if i < t0:
            continue
        # the Sobel gate reads image texture, so it always sees BGR
        sob = sobel_edges(np.ascontiguousarray(bgr, dtype=np.float32))
        gt = cv2.imread(os.path.join(root, "groundtruth", f"gt{i:06d}.png"), 0)
        for name, fn in chains.items():
            t = time.perf_counter()
            out = fn(np.asarray(mask), np.asarray(bg_prob), sob)
            secs[name] += time.perf_counter() - t
            acc[name] += confusion(out, gt, roi)
            empty[name] += not (out == 255).any()
        scored += 1

    print(f"\nhighway {t0}-{t1} ({scored} frames), model input = {colorspace}")
    print(f"  {'chain':36s} {'F1':>7s} {'IoU':>7s} {'P':>7s} {'R':>7s} "
          f"{'empty':>6s} {'ms/f':>7s}")
    rows = []
    for name, (tp, fp, fn_) in acc.items():
        f1, iou, p, r = metrics(tp, fp, fn_)
        rows.append((f1, iou, p, r, empty[name],
                     secs[name] / max(scored, 1) * 1000, name))
    for f1, iou, p, r, e, ms, name in sorted(rows, reverse=True):
        print(f"  {name:36s} {f1:7.4f} {iou:7.4f} {p:7.4f} {r:7.4f} "
              f"{e:6d} {ms:7.2f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--colorspace", default="both", choices=("bgr", "ycrcb", "both"))
    ap.add_argument("--last-frame", type=int, default=T1)
    ap.add_argument("--model", default="numba", choices=sorted(MODELS),
                    help="which backend produces the raw mask; the post chains "
                         "under test are the same for all of them")
    ap.add_argument("--parity-vs", metavar="BACKEND",
                    help="compare --model against BACKEND frame by frame and "
                         "exit non-zero on any difference. This is the gate; "
                         "an unchanged F1 is not one.")
    args = ap.parse_args()

    if args.parity_vs:
        cs = "ycrcb" if args.colorspace == "both" else args.colorspace
        bad = parity(args.dir, args.model, args.parity_vs, cs,
                     t1=args.last_frame)
        raise SystemExit(0 if bad == 0 else 1)

    spaces = ("bgr", "ycrcb") if args.colorspace == "both" else (args.colorspace,)
    for cs in spaces:
        score(args.dir, cs, t1=args.last_frame, model_name=args.model)
