"""bg_prob as the foreground decision instead of MOG2's binary one. Sweep the threshold.

MOG2's own decision is "the first matching mode lies inside the background set".
bg_prob is the summed weight of *every* background mode that matched, which is a
graded quantity the kernel already computes and used to throw away. Thresholding it
is a different, tunable decision boundary over the same evidence.
"""
import os, sys
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.environ.get("HIGHWAY_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "highway"))
from gmm import GMM_CPU_NUMBA
from gmm.mog2_common import to_planar
from settings import MOG2_N_COMPONENTS
from utils.post_processing import fill_holes

EL = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)


def variants(mask, bp):
    out = {}
    b = np.where(mask == 255, np.uint8(255), np.uint8(0))
    m = cv2.medianBlur(b, 5)
    out['mog2 decision +fill'] = fill_holes(m)
    out['mog2 decision +CLOSE15+fill'] = fill_holes(
        cv2.morphologyEx(m, cv2.MORPH_CLOSE, EL(15), iterations=1))
    for t in THRESHOLDS:
        mm = cv2.medianBlur(np.where(bp < t, np.uint8(255), np.uint8(0)), 5)
        out[f'bg_prob<{t} +fill'] = fill_holes(mm)
        out[f'bg_prob<{t} +CLOSE15+fill'] = fill_holes(
            cv2.morphologyEx(mm, cv2.MORPH_CLOSE, EL(15), iterations=1))
    return out


def run(cons):
    roi = cv2.imread(os.path.join(D, 'ROI.bmp'), 0) > 0
    first = cv2.imread(os.path.join(D, 'input', 'in000001.jpg'))
    cvt = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
    model = GMM_CPU_NUMBA(cvt(first), n_components=MOG2_N_COMPONENTS, conservative=cons)
    acc, empty, n = {}, {}, 0
    for i in range(1, 1701):
        bgr = cv2.imread(os.path.join(D, 'input', f'in{i:06d}.jpg'))
        mask, _ = model.step(to_planar(cvt(bgr)))
        if i < 470:
            continue
        gt = cv2.imread(os.path.join(D, 'groundtruth', f'gt{i:06d}.png'), 0)
        valid = roi & ((gt == 255) | (gt == 0)); g = (gt == 255) & valid
        for name, out in variants(np.asarray(mask), np.asarray(model.bg_prob)).items():
            p = (out == 255) & valid
            acc.setdefault(name, np.zeros(3))
            acc[name] += [np.sum(p & g), np.sum(p & ~g), np.sum(~p & g)]
            empty[name] = empty.get(name, 0) + (not (out == 255).any())
        n += 1
    print(f"\nhighway 470-1700 ({n} frames), conservative={cons}")
    print(f"  {'candidate':32s} {'F1':>7s} {'IoU':>7s} {'P':>7s} {'R':>7s} {'empty':>6s}")
    rows = []
    for name, (tp, fp, fn) in acc.items():
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        rows.append((2 * p * r / max(p + r, 1e-9), tp / max(tp + fp + fn, 1), p, r, empty[name], name))
    for f1, iou, p, r, e, name in sorted(rows, reverse=True):
        print(f"  {name:32s} {f1:7.4f} {iou:7.4f} {p:7.4f} {r:7.4f} {e:6d}")


if __name__ == '__main__':
    run(False)
    run(True)
