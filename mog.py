import sys

import cv2
import numpy as np

from utils import mask_refiner

# 1. Create the background subtractor object
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# 2. Open the video source: a path on the command line, else the default camera.
source = sys.argv[1] if len(sys.argv) > 1 else 0
capture = cv2.VideoCapture(source)
if not capture.isOpened():
    raise SystemExit(f"Cannot open {source!r}. Pass a video file as the first argument.")


dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

HOLD_FRAMES = 15          # ~0.5 s at 30 FPS
last_mask, held = None, 0

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # 3. Apply the subtractor to get the foreground mask
    fg_mask = back_sub.apply(frame)

    # 4. (Optional) Post-process to remove small noise points
    clean_mask = mask_refiner(fg_mask)

    # Sized and typed from the frame we are actually holding, and rebuilt every
    # iteration: keeping the previous frame's ellipse around leaves a patch of
    # background sharp after the subject has left it.
    elip_mask = np.zeros_like(clean_mask)

    # Find all white pixel coordinates
    pts = cv2.findNonZero(clean_mask)

    if pts is not None:
            # Reshape to (N, 2) -> columns are [X, Y]
        coords = pts.reshape(-1, 2)
        
        # 2. Calculate the Median Center (robust against outliers)
        median_center = np.median(coords, axis=0)
        
        # 3. Calculate distance of each pixel from the median
        distances = np.linalg.norm(coords - median_center, axis=1)
        
        # 4. Filter out outliers using MAD threshold
        # Multiply by 1.4826 to scale it like standard deviation
        mad = np.median(distances)
        threshold = 2.5 * (mad * 1.4826) 
        
        # Keep only inlier pixels
        inliers = coords[distances < threshold]

        # 5. Compute the final clean average coordinate
        if len(inliers) > 240:
            # 1. Calculate absolute horizontal (X) and vertical (Y) distances from the center
            inlier_centers = np.mean(inliers, axis=0) # Or use the median_center
            abs_x_distances = np.abs(inliers[:, 0] - inlier_centers[0])
            abs_y_distances = np.abs(inliers[:, 1] - inlier_centers[1])

            # 2. Get the maximum distances to serve as the ellipse radii
            max_x_dist = int(np.max(abs_x_distances))
            max_y_dist = int(np.max(abs_y_distances))

            # 3. Draw the ellipse mask safely
            center_coordinates = (int(inlier_centers[0]), int(inlier_centers[1]))
            axes_lengths = (max_x_dist, max_y_dist)

            cv2.ellipse(
                elip_mask, 
                center_coordinates, 
                axes_lengths, 
                angle=0, 
                startAngle=0, 
                endAngle=360, 
                color=255, 
                thickness=cv2.FILLED
            )

    # When the heuristic finds nothing, elip_mask is all zeros and copyTo blurs
    # the entire frame — the subject vanishes. Measured on input.mp4: 13 of 300
    # frames. Hold the last ellipse instead, for a bounded number of frames, so
    # a momentary miss does not wipe the picture but a subject who has actually
    # left does eventually stop being cut out.
    if elip_mask.any():
        last_mask, held = elip_mask, 0
    elif last_mask is not None and held < HOLD_FRAMES:
        elip_mask, held = last_mask, held + 1

    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    result = cv2.copyTo(frame, elip_mask, blur)

    # 5. Display the results
    cv2.imshow('Original Frame', frame)
    cv2.imshow("Clean mask", clean_mask)
    cv2.imshow('Foreground Mask', elip_mask)
    cv2.imshow("Result", result)

    # Exit if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
