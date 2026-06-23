import cv2
import numpy as np
import cupy as cp

from utils.timer import cpu_timer, gpu_timer

from gmm.cpu.GMM_cpu import GMM_CPU

cam = cv2.VideoCapture(0)
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

    mask = model.step(frame)


    if not flag:
        running = False
        continue
    
    cv2.imshow("Facecam", frame)

    cv2.imshow("Mask", (~mask))

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        running = False
