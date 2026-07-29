import numpy as np

def compute_iou(prediction: np.ndarray, groundtruth: np.ndarray):
        intersection = np.logical_and(prediction, groundtruth).sum()
        union = np.logical_or(prediction, groundtruth).sum()
        return intersection / union if union > 0 else 0.0