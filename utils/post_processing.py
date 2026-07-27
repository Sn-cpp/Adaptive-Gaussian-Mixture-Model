import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Post-processing initialization 

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


# --------------------------------------------------------------------------------------

def mask_refiner(mask: np.ndarray):

    # Use median blur to filter noise
    cleaned_mask = cv2.medianBlur(mask, 5)

    # Use morphology to refine the mask
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    cleaned_mask = cv2.dilate(cleaned_mask, kernel_dilate, iterations=1)

    return cleaned_mask

def background_subtractor(frame: np.ndarray, mask: np.ndarray):

    # Blur the entire frame, serving as background
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    result = cv2.copyTo(frame, foreground, blurred_frame)

    return result
