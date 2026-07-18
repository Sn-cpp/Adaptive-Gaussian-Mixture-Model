import cv2
import numpy as np
import cupy as cp

from utils.timer import cpu_timer, gpu_timer

from gmm import GMM_CPU, GMM_CPU_NUMBA

cam = cv2.VideoCapture('output_sequence.mp4')
# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_BUFFERSIZE, 9)

CAM_WIDTH = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
CAM_HEIGHT = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

running = True if cam.isOpened() else False

while True:
    flag, first_frame = cam.read()
    if flag:
        break

model = GMM_CPU_NUMBA(first_frame, n_components=7, parallel=True)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

while running:
    flag, frame = cam.read()

    if not flag:
        running = False
        continue

    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

    mask, time_cost = cpu_timer(model.step, frame, match_threshold=3.5, background_threshold=0.7, update_alpha=0.001)

    cleaned_mask = cv2.medianBlur(mask, 5)

    # frame = cv2.putText(frame, str(int(1/time_cost)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # cv2.imshow("Facecam", frame)

    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cleaned_mask = cv2.dilate(cleaned_mask, kernel_dilate, iterations=1)

    foreground = cv2.bitwise_and(frame, frame, mask=cleaned_mask)

    result = cv2.copyTo(frame, foreground, blurred_frame)

    cv2.imshow("Mask", cleaned_mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        running = False
