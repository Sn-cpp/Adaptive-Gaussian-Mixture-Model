"""Does a clean plate + protected update actually hold a stationary subject?

Ground truth has to be exact to answer that, so the subject is composited:
  background  real highway frames (static camera, real sensor noise)
  subject     a real crop of the person from LTSSUD-Test.mp4, pasted through
              an ellipse, entering at frame ENTER and then held perfectly still
Everything the model sees is real imagery; only the compositing is synthetic,
and that is exactly what buys a pixel-exact mask to score against.

Three configurations, all seeing the same frames:
  present-from-0    subject pasted from frame 0 -- the LTSSUD-Test.mp4 case
  enters-later      subject appears at ENTER, model saw the empty scene first
  + conservative    same, with the protected update on
"""
import os, sys
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gmm import GMM_CPU_NUMBA
from gmm.mog2_common import to_planar
from settings import MOG2_N_COMPONENTS
from utils.post_processing import mask_refiner

N = 260
ENTER = 60
BOX = (96, 60, 130, 150)      # x, y, w, h of the subject in a 320x240 frame


def subject():
    """A real person crop + its ellipse mask, sized to BOX."""
    cap = cv2.VideoCapture(os.path.join(ROOT, "LTSSUD-Test.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ok, f = cap.read()
    cap.release()
    assert ok
    x, y, w, h = BOX
    crop = cv2.resize(f[80:1000, 700:1620], (w, h))
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, 255, -1)
    return crop, m


def sequence(present_from_zero, noise=2.5, seed=0):
    """N frames of highway with the subject pasted in. Yields (frame, gt).

    `noise` is per-frame Gaussian noise on the pasted region only, in grey
    levels. Without it the paste is bit-identical every frame, which makes a
    motionless subject *literally* invisible to any background model and
    overstates the failure. Real webcam sensor noise at this exposure measures
    around 2-3 grey levels, so that is what goes on.
    """
    cap = cv2.VideoCapture(os.path.join(ROOT, "input.mp4"))
    crop, m = subject()
    x, y, w, h = BOX
    rng = np.random.default_rng(seed)
    for i in range(N):
        ok, bg = cap.read()
        if not ok:
            break
        bg = bg.copy()
        gt = np.zeros(bg.shape[:2], np.uint8)
        if present_from_zero or i >= ENTER:
            noisy = np.clip(crop + rng.normal(0, noise, crop.shape), 0, 255).astype(np.uint8)
            roi = bg[y:y + h, x:x + w]
            bg[y:y + h, x:x + w] = np.where(m[..., None] > 0, noisy, roi)
            gt[y:y + h, x:x + w] = m
        yield bg, gt
    cap.release()


def run(present_from_zero, cons, label, keep=()):
    cvt = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
    model = None
    ious, shots = [], {}
    for i, (frame, gt) in enumerate(sequence(present_from_zero)):
        if model is None:
            model = GMM_CPU_NUMBA(cvt(frame), n_components=MOG2_N_COMPONENTS,
                                  conservative=cons)
        mask, _ = model.step(to_planar(cvt(frame)))
        out = mask_refiner(np.asarray(mask))
        # score the subject only: the highway cars are real moving foreground
        # and detecting them is correct, so they are excluded rather than
        # counted as false positives.
        x, y, w, h = BOX
        p = (out[y:y + h, x:x + w] == 255)
        g = (gt[y:y + h, x:x + w] == 255)
        inter, union = (p & g).sum(), (p | g).sum()
        if i >= ENTER:
            ious.append(inter / max(union, 1))
        if i in keep:
            shots[i] = out.copy()
    a = np.array(ious)
    # last 100 frames = the subject has been motionless for four seconds
    print(f"  {label:32s} IoU {a.mean():.3f}  first20 {a[:20].mean():.3f}  "
          f"last100 {a[-100:].mean():.3f}  lost(<0.2) {int((a < 0.2).sum()):3d}/{len(a)}",
          flush=True)
    return shots


if __name__ == '__main__':
    print(f"composited highway + person, {N} frames, subject enters at {ENTER}")
    run(True, False, "present-from-0, plain MOG2")
    run(True, True, "present-from-0, conservative")
    run(False, False, "clean plate, plain MOG2")
    run(False, True, "clean plate + conservative")
