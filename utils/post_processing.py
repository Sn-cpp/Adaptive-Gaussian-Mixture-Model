"""Host-side post-processing — the specification the CUDA kernels must match.

The chain here is what `eval_highway.py` scored best on CDnet `highway`, the
full 470-1700 window, model input YCrCb:

    chain                                 F1      IoU       P       R   empty
    bg_prob < 0.5 + median5 + fill      0.9843  0.9691  0.9863  0.9823    0
    bg_prob < 0.5 + median5             0.9805  0.9617  0.9863  0.9748    0
    + CLOSE 15                          0.9782  0.9572  0.9634  0.9934    0
    Sobel gate + median3 + fill         0.9764  0.9538  0.9744  0.9783    0
    raw mask + median5 + fill           0.9111  0.8367  0.9971  0.8388    0
    Sobel gate + median3 + contour fill 0.7131  0.5541  0.5541  0.9999    0

Three things that table decided, all of which had been assumed the other way:

* **YCrCb, not BGR.** The identical chain scores 0.8272 on BGR input. Separating
  luma from chroma is the single largest factor in the whole pipeline, bigger
  than any post-processing step.
* **No CLOSE.** Its brush has to be small against the *object*, and a car is
  about 20 px in a 240 px frame, so even a 15 px brush is most of a car.
* **No contour fill.** Filling every external contour solid is right for one
  person and wrong for traffic: it swallows the road between cars, taking
  precision to 0.55 while recall goes to 1.0.

`fill_holes` is the only step that stays on the host in the GPU pipeline.
OpenCV implements it as a scan-line flood fill, which is sequential; the
data-parallel formulation is morphological reconstruction by dilation, so the
question is whether moving it is worth it rather than whether it is possible.
`bench_fill.py` implements both, checks they agree pixel-for-pixel, and times
them: at 1080p the reconstruction needs 593 full-frame dilate passes and 748 ms
against 2.6 ms. Each pass is a grid-wide step, and no amount of GPU shortens
the sequence of them. Profiling says leave it on the CPU, so we do.
"""
import cv2
import numpy as np

from settings import MOG2_BG_PROB_THRESHOLD


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every background region fully enclosed by foreground.

    Flood the background inward from the border, then OR back the complement:
    whatever the flood could not reach is a hole. Unlike a morphological CLOSE
    this fills a hole of any size or shape without moving the silhouette, and
    it cannot bridge two separate objects — which is exactly what a CLOSE wide
    enough to matter does to two cars in adjacent lanes.

    The flood starts from a one-pixel border added around the frame, not from
    pixel (0, 0). Seeding at the corner only works while that corner happens to
    be background: when an object touches it the flood cannot start, nothing is
    reachable, the complement is the whole image, and every pixel is declared
    foreground.
    """
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(padded)[1:h + 1, 1:w + 1]
    return mask | holes


def threshold_bg_prob(bg_prob: np.ndarray,
                      thresh=MOG2_BG_PROB_THRESHOLD) -> np.ndarray:
    """Foreground where the matched background modes carry too little weight.

    MOG2's own decision is the degenerate case `bg_prob > 0`: any match counts,
    however rarely that colour was seen. Asking for half the weight instead is
    worth +7 F1 here (0.9111 -> 0.9843) and costs one comparison per pixel.
    """
    return np.where(bg_prob < thresh, np.uint8(255), np.uint8(0))


def refine_mask(mask: np.ndarray, bg_prob: np.ndarray = None,
                close_ksize: int = 0, do_fill: bool = True) -> np.ndarray:
    """The shipping chain, and the reference the CUDA path is verified against.

    `bg_prob=None` falls back to MOG2's binary decision. `close_ksize > 1`
    inserts a CLOSE before the fill — measured as a loss on highway, kept for
    footage where the subject is large relative to the frame.
    """
    if bg_prob is not None:
        out = threshold_bg_prob(bg_prob)
    else:
        out = np.where(mask == 255, np.uint8(255), np.uint8(0))

    out = cv2.medianBlur(out, 5)
    if close_ksize > 1:
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, el, iterations=1)
    return fill_holes(out) if do_fill else out


def background_blur(frame_bgr: np.ndarray, mask: np.ndarray,
                    ksize: int = 15, sigma: float = 5.0) -> np.ndarray:
    """Keep the masked foreground sharp, blur everything else.

    `copyTo` wants a single-channel mask. Passing a 3-channel image makes
    OpenCV mask each channel separately, so any channel that happens to be 0 on
    the subject keeps the blurred value instead of the original one.
    """
    blurred = cv2.GaussianBlur(frame_bgr, (ksize, ksize), sigma)
    return cv2.copyTo(frame_bgr, mask, blurred)
