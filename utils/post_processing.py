import cv2
import numpy as np

# --------------------------------------------------------------------------------------

def fill_holes(mask: np.ndarray):
    """Fill every region of background fully enclosed by foreground.

    Flood the background inward from the image border; whatever the flood cannot
    reach is a hole, so OR its complement back in. Unlike a morphological CLOSE
    this fills a hole of any size and shape without growing the silhouette, and
    it cannot bridge two separate objects.

    The flood is sequential. A data-parallel equivalent exists (morphological
    reconstruction of the border marker) but it needs one dilate per pixel of
    propagation distance -- 512 iterations at 1080p, measured at 344 ms against
    2.2 ms for this -- so the scan-line flood is the right tool even in a
    GPU pipeline.
    """
    h, w = mask.shape
    flooded = mask.copy()
    cv2.floodFill(flooded, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return mask | cv2.bitwise_not(flooded)


def mask_refiner(mask: np.ndarray):
    """Binarise, despeckle, then close the holes MOG2 leaves inside an object.

    MOG2 writes 127 for shadow when `MOG2_DETECT_SHADOWS` is on, and everything
    downstream — medianBlur, floodFill, cv2.copyTo — treats any non-zero value
    as foreground, so a shadow would be kept sharp. Binarising here rather than
    at the call site means turning shadow detection back on cannot silently
    change what the blur composite does.

    Measured on CDnet `highway`, the full standard evaluation window (frames
    470-1700, 1231 scored frames), YCrCb input, GMM_CPU_NUMBA, scoring only
    ground-truth 0/255 pixels inside the ROI:

                                    F1     IoU      P       R    holes/f  empty
        raw MOG2                  0.8607  0.7554  0.9032  0.8220   77.2     0
        OPEN + CLOSE x2 + dilate  0.8748  0.7775  0.8121  0.9480    0.0     6
        medianBlur + fill_holes   0.9344  0.8769  0.9873  0.8869    0.0     0

    The morphology chain did remove the holes, but by inflating the silhouette
    until it swallowed them along with the shadow: precision 0.90 -> 0.81. It
    also produced 6 frames whose mask was *entirely empty* — for a blur product
    that means the subject vanishes for a moment, which is far worse than the
    F1 gap suggests. Filling holes directly keeps precision (0.90 -> 0.99) and
    never empties the mask, at 2.8 ms/frame at 1080p against 4.1 ms.

    Caveat worth keeping in view: `highway` is small high-contrast cars on grey
    asphalt, and it is also what `input.mp4` in this repo contains. No person or
    webcam footage has been scored anywhere in this project, so these numbers
    describe traffic, not the target application.
    """
    foreground = np.where(mask == 255, np.uint8(255), np.uint8(0))
    return fill_holes(cv2.medianBlur(foreground, 5))

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
