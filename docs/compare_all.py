"""Every post-processing candidate on one harness: ours, CLOSE, and Tin's two ideas.

Tin's main_contour.py (branch push_relabel_remade) proposes:
  fill_largest_contour  dilate 37 -> largest external contour -> fillPoly -> erode 13
  soft_alpha_composite  use bg_prob as a continuous alpha instead of a binary mask
Both are scored here against the same ground truth as everything else.
"""
import os, sys, time
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.environ.get("HIGHWAY_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "highway"))
from gmm import GMM_CPU_NUMBA
from gmm.mog2_common import to_planar
from settings import MOG2_N_COMPONENTS
from utils.post_processing import fill_holes

EL = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
RE = lambda k: cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


def fill_largest_contour(m, dilate_r=18, erode_r=6):
    """Tin's, verbatim in behaviour."""
    dil = cv2.dilate(m, EL(dilate_r * 2 + 1))
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(m)
    out = np.zeros_like(m)
    cv2.fillPoly(out, [max(cnts, key=cv2.contourArea)], 255)
    return cv2.erode(out, EL(erode_r * 2 + 1))


def fill_all_contours(m, dilate_r=18, erode_r=6):
    """Same, but keep every contour — the classroom clip has three people."""
    dil = cv2.dilate(m, EL(dilate_r * 2 + 1))
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        cv2.fillPoly(out, list(cnts), 255)
    return cv2.erode(out, EL(erode_r * 2 + 1))


def candidates(mask, bg_prob, k):
    b = np.where(mask == 255, np.uint8(255), np.uint8(0))
    m = cv2.medianBlur(b, 5)
    return {
        'median+fill (SHIPPING)': fill_holes(m),
        f'median+CLOSE{k}el+fill': fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, EL(k), iterations=1)),
        f'median+CLOSE{k}re+fill': fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, RE(k), iterations=1)),
        'Tin largest contour': fill_largest_contour(m),
        'Tin contour, all blobs': fill_all_contours(m),
        'bg_prob < 0.5': np.where(bg_prob < 0.5, np.uint8(255), np.uint8(0)),
        'bg_prob<0.5 +median+fill': fill_holes(cv2.medianBlur(
            np.where(bg_prob < 0.5, np.uint8(255), np.uint8(0)), 5)),
    }


def main(k=15):
    roi = cv2.imread(os.path.join(D, 'ROI.bmp'), 0) > 0
    first = cv2.imread(os.path.join(D, 'input', 'in000001.jpg'))
    cvt = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
    model = GMM_CPU_NUMBA(cvt(first), n_components=MOG2_N_COMPONENTS)
    acc, empty, secs, n = {}, {}, {}, 0
    for i in range(1, 1701):
        bgr = cv2.imread(os.path.join(D, 'input', f'in{i:06d}.jpg'))
        mask, _ = model.step(to_planar(cvt(bgr)))
        if i < 470:
            continue
        gt = cv2.imread(os.path.join(D, 'groundtruth', f'gt{i:06d}.png'), 0)
        valid = roi & ((gt == 255) | (gt == 0)); g = (gt == 255) & valid
        bp = np.asarray(model.bg_prob)
        for name, out in candidates(np.asarray(mask), bp, k).items():
            p = (out == 255) & valid
            acc.setdefault(name, np.zeros(3))
            acc[name] += [np.sum(p & g), np.sum(p & ~g), np.sum(~p & g)]
            empty[name] = empty.get(name, 0) + (not (out == 255).any())
        n += 1
    print(f"highway 470-1700 ({n} frames), conservative off")
    print(f"  {'candidate':26s} {'F1':>7s} {'IoU':>7s} {'P':>7s} {'R':>7s} {'empty':>6s}")
    for name, (tp, fp, fn) in acc.items():
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print(f"  {name:26s} {2*p*r/max(p+r,1e-9):7.4f} {tp/max(tp+fp+fn,1):7.4f} "
              f"{p:7.4f} {r:7.4f} {empty[name]:6d}")


if __name__ == '__main__':
    main()
