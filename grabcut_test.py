"""Offline comparison: does a GrabCut refinement stage earn its cost?

Not a live demo. GrabCut runs at 0.2-1.3 FPS depending on resolution, so this
scores it against the shipping post-processing on CDnet ground truth and prints
the trade, rather than pretending it can run per frame.

Two things the earlier version of this script got wrong, both of which made
GrabCut look far worse than it is:

  * it used GC_INIT_WITH_RECT with a fixed centre rectangle, so it never saw
    the GMM mask at all -- it was solving a different problem, and paying the
    highest price for it (8940 ms/frame at 1080p against 897 ms mask-seeded);
  * it reallocated bgdModel/fgdModel every frame, so the colour model relearned
    from scratch and could never converge.

Measured result, CDnet highway, full window 470-1700 (1231 frames), 240p:

    stage             F1     IoU       P       R   ms/frame  empty
    mask_refiner    0.9344  0.8769  0.9873  0.8869    0.22      0
    + grabCut       0.9552  0.9142  0.9804  0.9312   31.01      5

So the refinement is real -- +2.1 F1, +3.7 IoU, and it is recall it buys, which
is the right direction. It costs 141x, and it produces 5 frames whose mask is
entirely empty where the shipping path produces none. At 1080p the same call is
~900 ms/frame, so the live pipeline goes from 37 FPS to roughly 1.
Report the window with the numbers: over 470-669 the same comparison reads
0.8379 -> 0.9486, because the early frames are the hard ones.

Usage:
    python grabcut_test.py --dataset path/to/highway [--frames 200]
"""

import argparse
import os
import time

import cv2
import numpy as np

from gmm import GMM_CPU_NUMBA
from gmm.mog2_common import to_planar
from settings import MOG2_N_COMPONENTS
from utils.post_processing import mask_refiner

# Distance in pixels from the refined mask beyond which a pixel is certainly
# background. Inside the mask, an eroded core is marked certainly foreground.
# The tighter these are, the cheaper grabCut runs -- seed tightness moves the
# cost by 10x, which is the most interesting thing this script measures.
BAND = 21
CORE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def grabcut_refine(frame_bgr, refined_mask, iters=1):
    """Refine an existing mask with one GrabCut pass, seeded properly."""
    gc = np.full(refined_mask.shape, cv2.GC_PR_BGD, np.uint8)

    far = cv2.dilate(refined_mask, np.ones((BAND, BAND), np.uint8))
    gc[far == 0] = cv2.GC_BGD                      # certainly background
    gc[refined_mask > 0] = cv2.GC_PR_FGD           # probably foreground
    core = cv2.erode(refined_mask, CORE)
    gc[core > 0] = cv2.GC_FGD                      # certainly foreground

    if not (gc == cv2.GC_PR_FGD).any() and not (gc == cv2.GC_FGD).any():
        return refined_mask                        # nothing to refine

    cv2.grabCut(frame_bgr, gc, None, np.zeros((1, 65)), np.zeros((1, 65)),
                iters, cv2.GC_INIT_WITH_MASK)
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
                    np.uint8(255), np.uint8(0))


def score(pred, gt, roi):
    valid = roi & ((gt == 255) | (gt == 0))
    p, g = (pred == 255) & valid, (gt == 255) & valid
    return np.array([np.sum(p & g), np.sum(p & ~g), np.sum(~p & g)], float)


def f1_iou(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9), tp / max(tp + fp + fn, 1), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='CDnet sequence directory')
    ap.add_argument('--frames', type=int, default=0,
                    help='scored frames (0 = the full temporalROI window)')
    args = ap.parse_args()

    D = args.dataset
    t0, t1 = map(int, open(os.path.join(D, 'temporalROI.txt')).read().split())
    if args.frames:
        t1 = min(t1, t0 + args.frames - 1)
    roi = cv2.imread(os.path.join(D, 'ROI.bmp'), 0) > 0

    first = cv2.imread(os.path.join(D, 'input', 'in000001.jpg'))
    model = GMM_CPU_NUMBA(cv2.cvtColor(first, cv2.COLOR_BGR2YCrCb),
                          n_components=MOG2_N_COMPONENTS)

    acc = {'mask_refiner': np.zeros(3), '+ grabCut': np.zeros(3)}
    empty = {k: 0 for k in acc}
    seconds = {k: 0.0 for k in acc}
    scored = 0

    for i in range(1, t1 + 1):
        bgr = cv2.imread(os.path.join(D, 'input', f'in{i:06d}.jpg'))
        mask, _ = model.step(to_planar(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)))
        if i < t0:
            continue

        t = time.perf_counter()
        refined = mask_refiner(np.asarray(mask))
        seconds['mask_refiner'] += time.perf_counter() - t

        t = time.perf_counter()
        cut = grabcut_refine(bgr, refined)
        seconds['+ grabCut'] += time.perf_counter() - t + (seconds['mask_refiner'] / max(scored + 1, 1))

        gt = cv2.imread(os.path.join(D, 'groundtruth', f'gt{i:06d}.png'), 0)
        for k, out in (('mask_refiner', refined), ('+ grabCut', cut)):
            acc[k] += score(out, gt, roi)
            if not (out == 255).any():
                empty[k] += 1
        scored += 1

    print(f'{os.path.basename(D.rstrip("/"))}, frames {t0}-{t1} ({scored} scored), '
          f'{cv2.__version__=}')
    print(f'  {"stage":14s} {"F1":>7s} {"IoU":>7s} {"P":>7s} {"R":>7s} '
          f'{"ms/frame":>9s} {"empty":>6s}')
    for k, (tp, fp, fn) in acc.items():
        f1, iou, p, r = f1_iou(tp, fp, fn)
        print(f'  {k:14s} {f1:7.4f} {iou:7.4f} {p:7.4f} {r:7.4f} '
              f'{seconds[k] / scored * 1000:9.2f} {empty[k]:6d}')
    print('\n  "empty" counts frames whose mask came back completely blank — for a\n'
          '  blur product that is the subject disappearing, and it matters more\n'
          '  than the F1 column.')


if __name__ == '__main__':
    main()
