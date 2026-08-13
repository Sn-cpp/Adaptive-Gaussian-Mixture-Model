import cv2
import numpy as np

from settings import (CLOSE_KSIZE_FRACTION, CLOSE_KSIZE_MAX,
                      MOG2_BG_PROB_THRESHOLD)

# --------------------------------------------------------------------------------------

def fill_holes(mask: np.ndarray):
    """Fill every region of background fully enclosed by foreground.

    Flood the background inward from the image border; whatever the flood cannot
    reach is a hole, so OR its complement back in. Unlike a morphological CLOSE
    this fills a hole of any size and shape without growing the silhouette, and
    it cannot bridge two separate objects.

    The flood starts from a one-pixel background border added around the frame,
    not from pixel (0, 0). Seeding at (0, 0) only works while that corner
    happens to be background: when the subject touches it — a shoulder in the
    top-left of a webcam frame — the flood cannot start, nothing is reachable,
    and the complement is the entire image, so the whole frame is declared
    foreground. Padding makes every border-adjacent background region reachable
    by construction, whatever the corner holds.

    The flood is sequential. A data-parallel equivalent exists (morphological
    reconstruction of the border marker) but it needs one dilate per pixel of
    propagation distance -- 512 iterations at 1080p, measured at 344 ms against
    2.2 ms for this -- so the scan-line flood is the right tool even in a
    GPU pipeline.
    """
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(padded)[1:h + 1, 1:w + 1]
    return mask | holes


def close_ksize_for(frame: np.ndarray):
    """CLOSE kernel scaled to the frame, odd, capped — see `CLOSE_KSIZE_FRACTION`.

    The gaps to be closed scale with the apparent size of the object, not with
    the pixel grid, so a kernel tuned at 240p is nearly useless at 1080p. This
    is linear in frame height, the same rule `blur_ksize_for` uses, which holds
    only while the framing is comparable — a webcam at arm's length and a
    traffic camera are not, and neither is a person who walks towards the lens.
    """
    k = int(round(CLOSE_KSIZE_FRACTION * frame.shape[0]))
    return max(3, min(k | 1, CLOSE_KSIZE_MAX))


def mask_refiner(mask: np.ndarray, bg_prob: np.ndarray = None,
                 close_ksize: int = 0):
    """Binarise, despeckle, then close the holes MOG2 leaves inside an object.

    MOG2 writes 127 for shadow when `MOG2_DETECT_SHADOWS` is on, and everything
    downstream — medianBlur, floodFill, cv2.copyTo — treats any non-zero value
    as foreground, so a shadow would be kept sharp. Binarising here rather than
    at the call site means turning shadow detection back on cannot silently
    change what the blur composite does.

    `bg_prob`, when given, replaces MOG2's binary decision with a threshold on
    the background confidence the kernel already computes. `close_ksize > 1`
    inserts a morphological CLOSE before the hole fill; use `close_ksize_for`
    to scale it to the frame.

    Measured on CDnet `highway`, the full standard evaluation window (frames
    470-1700, 1231 scored frames), YCrCb input, GMM_CPU_NUMBA, conservative
    update off, scoring only ground-truth 0/255 pixels inside the ROI:

                                    F1     IoU      P       R    empty
        raw MOG2                  0.8607  0.7554  0.9032  0.8220    0
        median only               0.9248  0.8602  0.9872  0.8699    0
        median + fill_holes       0.9338  0.8758  0.9873  0.8858    0
        + CLOSE 15 (ellipse)      0.9542  0.9123  0.9709  0.9379    0
        bg_prob<0.5, median, fill 0.9633  0.9292  0.9418  0.9857    0
        bg_prob<0.3, median, fill 0.9636  0.9297  0.9434  0.9847    0

    Two findings there, both worth more than they look.

    **bg_prob beats the binary decision by 3.0 F1, for free.** MOG2 calls a
    pixel background if the first mode it matches lies inside the background
    set — any match, however little weight that mode carries. `bg_prob` is the
    summed weight of every background mode that matched, so `background` in the
    kernel is exactly `bg_prob > 0`. Thresholding at 0.5 instead demands that
    the matched modes carry half the weight, which throws out matches against
    spurious low-weight modes. Recall 0.886 -> 0.986 at a precision cost of
    0.987 -> 0.942. The result is flat between 0.3 and 0.5, so it is not tuned
    to a knife edge, and the value was already being computed and discarded.
    For reference the GrabCut refinement in `graphcut/` scores F1 0.9552 at
    31 ms/frame; this beats it at the cost of one comparison per pixel.

    **A CLOSE helps; the OPEN was what was hurting.** The old
    `OPEN + CLOSE x2 + dilate` chain scored 0.8748 with 6 entirely empty masks,
    and that damage is not the CLOSE: `median + OPEN` alone scores 0.9182 and
    produces all 6 of those empty frames, while the final dilate is what drops
    precision to 0.81. A CLOSE on its own is the second-best thing measured
    here. This is why the pipelines' morphology was flipped from OPEN to CLOSE
    as well — see `utils.blur_numba.morph_close`.

    **The CLOSE is off unless the caller asks.** Its kernel has to be small
    against the *object*, and `close_ksize_for` can only scale it against the
    *frame*. On a webcam those are close enough — a seated person fills half
    the height — and the CLOSE is worth a great deal: on `LTSSUD-Test.mp4` at
    480x270 with conservative update on, `bg_prob` alone gives 19.4% mask
    coverage against a subject that really occupies 25-30%, with 29 of 310
    frames essentially empty and 55 connected components; adding CLOSE 17 gives
    27.8% coverage, 10 empty frames and 10 components. On `highway` those same
    two scales are nothing alike — a car is ~20 px in a 240 px frame, so a
    15 px kernel is most of a car — and the CLOSE drops F1 from 0.9633 to
    0.9316. `main.py` turns it on because it is a webcam application; the CDnet
    scoring path leaves it off.

    A CLOSE wider than the gap between two objects merges them, and F1 hides
    it — a 15x15 CLOSE across a 10-pixel aisle between two people fills 98.6%
    of the aisle while F1 stays near 0.96. Keep `CLOSE_KSIZE_MAX` in view.

    Caveat worth keeping in view: `highway` is small high-contrast cars on grey
    asphalt, and it is also what `input.mp4` in this repo contains. These
    numbers describe traffic, not the target application — see
    `docs/conservative.md` for what happens on webcam footage, where the
    limiting factor is not post-processing at all.
    """
    if bg_prob is not None:
        foreground = np.where(bg_prob < MOG2_BG_PROB_THRESHOLD,
                              np.uint8(255), np.uint8(0))
    else:
        foreground = np.where(mask == 255, np.uint8(255), np.uint8(0))

    refined = cv2.medianBlur(foreground, 5)
    if close_ksize > 1:
        # RECT, not ELLIPSE: 0.9516 against 0.9542 on highway, but 3.6 ms
        # against 22.3 ms at 1080p with a 61-wide kernel, because OpenCV
        # decomposes a rectangle into two 1-D passes and cannot do that for an
        # ellipse. The same separability argument as the Gaussian blur.
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, el, iterations=1)
    return fill_holes(refined)

def blur_ksize_for(frame: np.ndarray, reference_height: int = 240,
                   reference_ksize: int = 15):
    """Scale the blur kernel with the frame, so the effect looks the same.

    A fixed 15x15 is a strong blur on a 240p face and barely visible on a 1080p
    one, because 15 pixels covers a much smaller share of the subject. Measured
    on LTSSUD-Test.mp4, residual sharpness (variance of Laplacian, as a fraction
    of the unblurred frame) after a fixed 15x15:

        240p 0.7%    480p 1.5%    720p 4.9%    1080p 8.1%

    Scaling by height keeps that constant. `settings.BLUR_KSIZE` stays 15 and
    still drives blur_numba / blur_cuda, whose benchmark figures were all taken
    at a fixed kernel; this only affects the composite the demo displays.
    """
    k = int(round(reference_ksize * frame.shape[0] / reference_height))
    return max(3, k | 1)


def background_subtractor(frame: np.ndarray, mask: np.ndarray, ksize: int = None):
    """Keep the masked subject sharp, blur everything else."""
    if ksize is None:
        ksize = blur_ksize_for(frame)

    blurred_frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

    # copyTo wants a binary mask, not an image. Passing a 3-channel BGR image
    # makes OpenCV mask each channel separately, so any channel that happens to
    # be 0 on the subject (dark hair, dark clothing, shadow) keeps the blurred
    # value instead of the original one.
    result = cv2.copyTo(frame, mask, blurred_frame)

    return result
