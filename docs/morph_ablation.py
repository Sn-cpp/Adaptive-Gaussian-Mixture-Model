"""Which morphology step helps, which empties the mask? Every op keyword-argumented.

Positional args bite here: cv2.morphologyEx(src, op, kernel, dst, anchor, iterations)
-- the 4th positional is `dst`, not `iterations`, so passing 2 there silently runs
one iteration. Everything below names `iterations=` explicitly.
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
k_open, k_dil = EL(5), EL(7)
close = lambda m, k, it=1: cv2.morphologyEx(m, cv2.MORPH_CLOSE, EL(k), iterations=it)
opn = lambda m: cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1)


def stages(mask):
    b = np.where(mask == 255, np.uint8(255), np.uint8(0))
    m = cv2.medianBlur(b, 5)
    out = {
        'median only': m,
        'median+fill (SHIPPING)': fill_holes(m),
        'median+OPEN': opn(m),
        'median+OPEN+CLOSE15x2': close(opn(m), 15, 2),
        'old chain (OPEN+CLOSE15x2+dil)': cv2.dilate(close(opn(m), 15, 2), k_dil, iterations=1),
    }
    for k in (5, 7, 9, 11, 15, 21):
        out[f'median+CLOSE{k}+fill'] = fill_holes(close(m, k))
    return out


def main():
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
        valid = roi & ((gt == 255) | (gt == 0))
        g = (gt == 255) & valid
        t0 = time.perf_counter()
        st = stages(np.asarray(mask))
        for k, out in st.items():
            p = (out == 255) & valid
            acc.setdefault(k, np.zeros(3))
            acc[k] += [np.sum(p & g), np.sum(p & ~g), np.sum(~p & g)]
            empty[k] = empty.get(k, 0) + (not (out == 255).any())
        n += 1
    print(f"highway 470-1700 ({n} frames), YCrCb, GMM_CPU_NUMBA")
    print(f"  {'stage':32s} {'F1':>7s} {'IoU':>7s} {'P':>7s} {'R':>7s} {'empty':>6s}")
    for k, (tp, fp, fn) in acc.items():
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print(f"  {k:32s} {2*p*r/max(p+r,1e-9):7.4f} {tp/max(tp+fp+fn,1):7.4f} "
              f"{p:7.4f} {r:7.4f} {empty[k]:6d}")


if __name__ == '__main__':
    main()
