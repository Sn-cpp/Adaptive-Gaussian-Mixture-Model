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


while running:
    flag, frame = cam.read()

    if not flag:
        running = False
        continue

    mask, update_cost, predict_cost = model.step_profiler(frame)


    # frame = cv2.putText(frame, str(int(1/time_cost)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # cv2.imshow("Facecam", frame)

    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        running = False

    break

update_df = pd.DataFrame.from_dict(update_cost, orient='index', columns=['Time (ms)'])
update_df['Time (ms)'] *= 1000
update_df['Time (ms)'].round(3)
print(update_df)


predict_df = pd.DataFrame.from_dict(predict_cost, orient='index', columns=['Time (ms)'])
predict_df['Time (ms)'] *= 1000
predict_df['Time (ms)'].round(3)
print(predict_df)

# print("Update phase:")
# for k, v in update_cost.items():
#     print(f'{k}\t{round(v * 1000, 2)} ms')

# print("\nPredict phase:")
# for k, v in predict_cost.items():
#     print(f'{k}\t{round(v * 1000, 2)} ms')


    
