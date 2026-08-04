import cv2
from utils import mask_refiner
import numpy as np

# 1. Create the background subtractor object
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# 2. Open the video source (0 for webcam, or path to a video file)
capture = cv2.VideoCapture(0)
# capture = cv2.VideoCapture("input.mp4")


dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

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
