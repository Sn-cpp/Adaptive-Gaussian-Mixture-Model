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
from utils.post_processing import (background_blur, fill_holes, refine_mask,
                                   threshold_bg_prob)

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
    """Per-frame comparison of two backends' shipping mask and composite.

    The verdict has three levels, because the pipeline has two kinds of stage:

    **Integer stages must be bit-exact, no exceptions.** The colour conversion,
    threshold-as-written, median, flood fill, blur and composite are integer
    arithmetic end to end; any difference there is a bug, and this gate fails
    hard on it.

    **The float32 model stage cannot promise bit-identity across compilers.**
    LLVM on the host and NVVM on the device contract fused-multiply-adds
    differently, so `bg_prob` can differ by ulps — and a pixel whose true value
    sits within an ulp of the 0.5 threshold flips. This is the same phenomenon
    as our measured 22-px disagreement with cv2's own MOG2, and the docs have
    said so since the blur kernels landed. On synthetic frames no pixel lands
    that close; long real sequences find a few.

    So the gate does not grant a blanket tolerance. The compared pre-fill mask
    is post-median, and the 5x5 median moves a flip to neighbouring pixels — so
    the diagnosis re-thresholds **both** backends' `bg_prob` fields to recover
    the raw pre-median masks, and every raw flip must be proven individually:
    the two values must straddle the threshold and differ by less than
    BOUNDARY_EPS. It then re-runs the median on each raw mask and requires it
    to reproduce that backend's pre-fill mask exactly, so the post-median
    differences are fully accounted for by the proven flips. Every flip proven
    → verdict FLOAT-BOUNDARY, exit 0, with the per-pixel evidence printed. Any
    flip that is *not* a proven boundary case — or any difference in an
    integer stage — is DIVERGED, exit 1.
    """
    BOUNDARY_EPS = 1e-4          # |bg_prob_a - bg_prob_b| at a flipped pixel

    first = cv2.imread(os.path.join(root, "input", "in000001.jpg"))
    if first is None:
        raise SystemExit(f"no frames under {root}/input — set HIGHWAY_DIR")
    h, w = first.shape[:2]

    def shipping_chain(name):
        """(model, run) with run(bgr) -> (pre_fill_mask, refined, composite),
        plus a bg_prob fetch for the boundary diagnosis."""
        m = build_model(name, h, w, post=True, colorspace=colorspace)
        state = {}
        if hasattr(m, "mask_from_bgr"):
            def run(bgr):
                pre = m.mask_from_bgr(bgr).copy()
                refined = fill_holes(pre)
                return pre, refined, m.composite(refined)
            state["bg"] = lambda: m.d_bg_prob.copy_to_host()
            return m, run, state

        cvt = ((lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb))
               if colorspace == "ycrcb" else (lambda f: f))

        def run(bgr):
            mask, bg_prob, _ = m.apply(cvt(bgr))
            state["_bg"] = np.asarray(bg_prob).copy()
            pre = refine_mask(np.asarray(mask), bg_prob=state["_bg"],
                              do_fill=False)
            refined = fill_holes(pre)
            return pre, refined, background_blur(bgr, refined,
                                                 BLUR_KSIZE, BLUR_SIGMA)
        state["bg"] = lambda: state["_bg"]
        return m, run, state

    _, run_a, st_a = shipping_chain(model_name)
    _, run_b, st_b = shipping_chain(ref_name)
    thr = float(MOG2_BG_PROB_THRESHOLD)

    n_pre = n_ref = n_comp = scored = 0
    integer_stage_bug = False
    flips = []                   # (frame, y, x, bg_a, bg_b, proven)
    saw_fg = False
    for i in range(1, t1 + 1):
        bgr = cv2.imread(os.path.join(root, "input", f"in{i:06d}.jpg"))
        pa, ra, ca = run_a(bgr)
        pb, rb, cb = run_b(bgr)
        if i < t0:
            continue
        scored += 1
        saw_fg |= bool((rb == 255).any())

        dp = int((pa != pb).sum())
        dr = int((np.asarray(ra) != np.asarray(rb)).sum())
        dc = int((np.asarray(ca) != np.asarray(cb)).sum())
        n_pre += dp; n_ref += dr; n_comp += dc

        if dp:
            ba, bb = st_a["bg"](), st_b["bg"]()
            ta, tb = threshold_bg_prob(ba), threshold_bg_prob(bb)
            for y, x in np.argwhere(ta != tb):
                va, vb = float(ba[y, x]), float(bb[y, x])
                proven = (abs(va - vb) < BOUNDARY_EPS and
                          (va - thr) * (vb - thr) <= 0)
                flips.append((i, int(y), int(x), va, vb, proven))
            # the raw flips must fully explain the post-median difference:
            # each backend's own 5x5 median over its raw mask has to
            # reproduce its pre-fill mask bit for bit
            if ((ta == tb).all()
                    or (cv2.medianBlur(ta, 5) != np.asarray(pa)).any()
                    or (cv2.medianBlur(tb, 5) != np.asarray(pb)).any()):
                integer_stage_bug = True
        if (dr and not dp) or (dc and not dr):
            # a difference appearing in an integer stage with identical input
            integer_stage_bug = True
        if dc:
            # zero tolerance on the composite even when the masks legitimately
            # differ: each backend's composite must equal the host blur over
            # its own refined mask (the Q8 blur is pinned bit-exact to cv2)
            for comp, ref in ((ca, ra), (cb, rb)):
                if (np.asarray(comp) != background_blur(
                        bgr, np.asarray(ref), BLUR_KSIZE, BLUR_SIGMA)).any():
                    integer_stage_bug = True

    px = scored * h * w
    print(f"\nparity {model_name} vs {ref_name}, {colorspace}, shipping mask "
          f"and composite, frames {t0}-{t1} ({scored} frames, {px} pixels each)")
    print(f"  pre-fill mask differing px : {n_pre}  ({n_pre / max(px, 1):.6%})")
    print(f"  refined mask differing px  : {n_ref}")
    print(f"  composite differing px     : {n_comp}")
    if flips:
        print(f"  flipped-pixel evidence (threshold {thr}):")
        for f, y, x, va, vb, ok in flips[:20]:
            print(f"    frame {f} ({y:3d},{x:3d})  bg_prob {va:.7f} vs {vb:.7f} "
                  f"|Δ|={abs(va-vb):.2e}  {'boundary ✓' if ok else 'NOT boundary ✗'}")
    if not saw_fg:
        print("  DEGENERATE — no foreground in any frame; proves nothing")
        return 1

    all_proven = flips and all(f[5] for f in flips)
    if n_pre == 0 and n_ref == 0 and n_comp == 0:
        print("  VERDICT: IDENTICAL")
        return 0
    if all_proven and not integer_stage_bug:
        dmax = max(abs(f[3] - f[4]) for f in flips)
        print(f"  VERDICT: FLOAT-BOUNDARY — {len(flips)} flips over {scored} "
              f"frames, every one a proven threshold-straddle (max |Δ| {dmax:.2e}).")
        print("  The integer stages are exact; this is the documented float32 "
              "compiler-contraction limit, same class as the 22-px cv2 residual.")
        return 0
    print("  VERDICT: DIVERGED — differences are not explained by the float32 "
          "threshold boundary. This is a bug, not rounding.")
    return 1


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
