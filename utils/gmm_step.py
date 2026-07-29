import numpy as np
import cupy as cp
from .timer import *

def cpu_step(model, frame: np.ndarray, match_threshold: np.float32, update_alpha: np.float32, weight_threshold: np.float32):
    # Predict step 
    (mask, diff_square_sum), predict_cost = cpu_timer(model.predict, frame=frame, match_threshold=match_threshold, weight_threshold=weight_threshold)

    # Update step
    _, update_cost = cpu_timer(model.update, frame=frame, diff_square_sum=diff_square_sum, match_threshold=match_threshold, update_alpha=update_alpha)

    return mask, predict_cost + update_cost

def gpu_step(model, frame: np.ndarray, match_threshold: np.float32, update_alpha: np.float32, weight_threshold: np.float32):
    # Data transfer from host to device
    gpu_frame, to_dvc_cost = cpu_timer(cp.asarray, frame)

    # Predict step
    (gpu_mask, diff_square_sum), predict_cost = gpu_timer(model.predict, frame=gpu_frame, match_threshold=match_threshold, weight_threshold=weight_threshold)


    # Update step
    _, update_cost = gpu_timer(model.update, frame=gpu_frame, diff_square_sum=diff_square_sum, match_threshold=match_threshold, update_alpha=update_alpha)

    # Data transfer from device to host
    mask, to_host_cost = cpu_timer(gpu_mask.get)

    total_time = to_dvc_cost + (predict_cost + update_cost) + to_host_cost
    return mask, total_time


