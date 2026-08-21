import cv2
import numpy as np

def to_planar(frame_bgr: np.ndarray, color=True):
    """(H, W, 3) uint8 BGR -> (C, H, W) float32 model input."""
    if color:
        return np.ascontiguousarray(frame_bgr.transpose(2, 0, 1).astype(np.float32))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(gray.astype(np.float32)[None])
