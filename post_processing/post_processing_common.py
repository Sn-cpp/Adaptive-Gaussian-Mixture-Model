import numpy as np
import cv2
from time import perf_counter

def gaussian_kernel_2d(kernel_size: int, sigma=1.0):
    """Generates a normalized 2D Gaussian kernel."""
    ax = np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
    gauss_1d = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel_2d = np.outer(gauss_1d, gauss_1d)
    return kernel_2d / np.sum(kernel_2d)

class PostProcessingBase:
    def __init__(self):
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        self.kernel_gauss = gaussian_kernel_2d(15)

        self.processed_frame = None

    def apply(self, frame: np.ndarray, mask: np.ndarray):
        t0 = perf_counter()
        self._apply_kernel(frame, mask)
        return self.processed_frame, perf_counter() - t0 

    def _apply_kernel(self, frame, mask, args):
        raise NotImplementedError