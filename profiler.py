import cv2
import numpy as np
import cupy as cp
import pandas as pd

from utils.timer import cpu_timer, gpu_timer

from gmm.cpu.GMM_cpu import GMM_CPU

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

CAM_WIDTH = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
CAM_HEIGHT = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

running = True if cam.isOpened() else False

while True:
    flag, first_frame = cam.read()
    if flag:
        break

model = GMM_CPU(first_frame, n_components=5)

num_frames = 5
avg_update = None
avg_predict = None

for i in range(num_frames):
    flag, frame = cam.read()

    if not flag:
        break

    mask, update_cost, predict_cost = model.step_profiler(frame)
    update_df = pd.DataFrame.from_dict(update_cost, orient='index', columns=['Time (ms)'])
    predict_df = pd.DataFrame.from_dict(predict_cost, orient='index', columns=['Time (ms)'])

    if i == 0:
        avg_update = update_df
        avg_predict = predict_df
        continue

    avg_update = avg_update + update_df
    avg_predict = avg_predict + predict_df

print("Prediction: ")
avg_predict['Time (ms)'] *= (1000 / num_frames)
avg_predict['Time (ms)'].round(3)
print(avg_predict)

print("\nUpdating")
avg_update['Time (ms)'] *= (1000 / num_frames)
avg_update['Time (ms)'].round(3)
print(avg_update)

# print("Update phase:")
# for k, v in update_cost.items():
#     print(f'{k}\t{round(v * 1000, 2)} ms')

# print("\nPredict phase:")
# for k, v in predict_cost.items():
#     print(f'{k}\t{round(v * 1000, 2)} ms')


    
