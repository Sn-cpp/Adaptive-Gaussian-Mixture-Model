import cv2
import numpy as np

# 1. Kết nối với Webcamq
cap = cv2.VideoCapture("footage.mp4")

if not cap.isOpened():
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
    
    h, w, _ = frame.shape


    rect_w, rect_h = int(w * 0.6), int(h * 0.7)
    rect_x = (w - rect_w) // 2
    rect_y = (h - rect_h) // 2
    rect = (rect_x, rect_y, rect_w, rect_h)

    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, iterCount=2, mode=cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_obj = frame * mask2[:, :, np.newaxis]

    bg = cv2.GaussianBlur(frame, (15, 15), 0.0)

    cv2.copyTo(frame, segmented_obj, bg)


    cv2.imshow("Mask", segmented_obj)
    cv2.imshow("GrabCut Real-time", bg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
