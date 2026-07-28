import cv2
import numpy as np
import cupy as cp

from utils import *

from gmm import *

from tester import compare_diff_square_sum

# --------------------------------------------------------------------------------------
# Input initialization

# cam = cv2.VideoCapture('test_video.mp4')
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
# 1 - Numba parallelized
# 2 - Cupy built-in functions on GPU 

model_list = [
    GMM_CPU,
    GMM_CPU_NUMBA,
    GMM_CUPY_V0,
    GMM_CUPY_V1
]

# Model selection
model_choice = 3

# Parameters config
gaussian_components = 7
match_threshold = np.float32(3.5)
background_threshold = np.float32(0.7)
update_alpha = np.float32(0.01)

model = model_list[model_choice](first_frame, n_components=gaussian_components, parallel=True)
model_base = GMM_CPU(first_frame, n_components=gaussian_components)


# --------------------------------------------------------------------------------------
# Utilities 

def cpu_step(model, frame: np.ndarray):
    # Predict step 
    (mask, diff_square_sum), predict_cost = cpu_timer(model.predict, frame=frame, match_threshold=match_threshold, background_threshold=background_threshold)

    # Update step
    _, update_cost = cpu_timer(model.update, frame=frame, diff_square_sum=diff_square_sum, match_threshold=match_threshold, update_alpha=update_alpha)

    return mask, predict_cost + update_cost

def gpu_step(model, frame: np.ndarray):
    # Data transfer from host to device
    gpu_frame, to_dvc_cost = cpu_timer(cp.asarray, frame)

    # Predict step
    (gpu_mask, diff_square_sum), predict_cost = gpu_timer(model.predict, frame=gpu_frame, match_threshold=match_threshold, background_threshold=background_threshold)


    # Update step
    _, update_cost = gpu_timer(model.update, frame=gpu_frame, diff_square_sum=diff_square_sum, match_threshold=match_threshold, update_alpha=update_alpha)

    # Data transfer from device to host
    mask, to_host_cost = cpu_timer(gpu_mask.get)

    total_time = to_dvc_cost + (predict_cost + update_cost) + to_host_cost
    return mask, total_time

model_fps_graph = FPS_Graph(400, 400)


# --------------------------------------------------------------------------------------
base_func = cpu_step
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

    # Convert the frame to planar mode (C, H, W) with C=3 (BGR)
    planar_frame = frame.transpose(2, 0, 1).astype(np.float32)

    mask, time_cost = step_func(model, planar_frame)
    model_fps = int(1/time_cost)
    model_fps_graph.write_value(model_fps)
    # mask_base, time_cost_base = base_func(model_base, planar_frame)

    refined_mask = mask_refiner(mask)
    result = background_subtractor(frame, refined_mask)
    # result = cv2.putText(result, str(int(1/time_cost)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)


    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    model_fps_graph.display("Numba FPS", model_fps)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        running = False
