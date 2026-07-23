import cv2
import numpy as np
import cupy as cp

from utils import cpu_timer, gpu_timer, cp_gpu_warmup

from gmm import *

# --------------------------------------------------------------------------------------
# Input initialization

# cam = cv2.VideoCapture('output_sequence.mp4')
cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_BUFFERSIZE, 9)

CAM_WIDTH = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
CAM_HEIGHT = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

running = True if cam.isOpened() else False

while True:
    flag, first_frame = cam.read()
    if flag:
        break


# --------------------------------------------------------------------------------------
# Model initialization
# 0 - CPU - Numpy baseline
# 1 - Numba optimized/parallelized
# 2 - Cupy built-in functions on GPU 

model_list = [
    GMM_CPU,
    GMM_CPU_NUMBA,
    GMM_CUPY_V0
]

# Model selection
model_choice = 2

# Parameters config
gaussian_components = 7
match_threshold = 3.5
background_threshold = 0.7
update_alpha = 0.001

model = model_list[model_choice](first_frame, n_components=7, parallel=True)

# --------------------------------------------------------------------------------------
# Post-processing initialization 

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


# --------------------------------------------------------------------------------------
# Utilities 

def cpu_step(frame: np.ndarray):
    # Predict step 
    (mask, diff_square_sum), predict_cost = cpu_timer(model.predict, frame=frame, match_threshold=match_threshold, background_threshold=background_threshold)

    # Update step
    _, update_cost = cpu_timer(model.update, frame=frame, diff_square_sum=diff_square_sum, match_threshold=match_threshold, update_alpha=update_alpha)

    return mask, predict_cost + update_cost

def gpu_step(frame: np.ndarray):
    # Data transfer from host to device
    gpu_frame, to_dvc_cost = cpu_timer(cp.asarray, frame)

    # Predict step
    (gpu_mask, diff_square_sum), predict_cost = gpu_timer(model.predict, frame=gpu_frame, match_threshold=match_threshold, background_threshold=background_threshold)

    # Update step
    _, update_cost = gpu_timer(model.update, frame=gpu_frame, diff_square_sum=diff_square_sum, match_threshold=match_threshold, update_alpha=update_alpha)

    # Data transfer from device to host
    mask, to_host_cost = cpu_timer(gpu_mask.get)

    total_time = to_dvc_cost + (predict_cost + update_cost) / 1000.0 + to_host_cost
    return mask, total_time


# --------------------------------------------------------------------------------------

if model_choice >= 2:
    step_func = gpu_step
    cp_gpu_warmup()
else:
    step_func = cpu_step


print("Ready")
while running:
    flag, frame = cam.read()

    if not flag:
        running = False
        continue

    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

    mask, time_cost = step_func(frame)
    print(time_cost)


    cleaned_mask = cv2.medianBlur(mask, 5)

    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cleaned_mask = cv2.dilate(cleaned_mask, kernel_dilate, iterations=1)


    foreground = cv2.bitwise_and(frame, frame, mask=cleaned_mask)

    result = cv2.copyTo(frame, foreground, blurred_frame)

    result = cv2.putText(result, str(int(1/time_cost)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Mask", cleaned_mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        running = False
