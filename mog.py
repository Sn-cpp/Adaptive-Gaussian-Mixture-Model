import cv2

# 1. Create the background subtractor object
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# 2. Open the video source (0 for webcam, or path to a video file)
capture = cv2.VideoCapture("input.mp4")

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # 3. Apply the subtractor to get the foreground mask
    fg_mask = back_sub.apply(frame)

    # 4. (Optional) Post-process to remove small noise points
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # 5. Display the results
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    # Exit if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
