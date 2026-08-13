from gmm_mask import warmup_mask_gmm_jit
warmup_mask_gmm_jit()

import argparse
import cv2
import numpy as np

from gmm_mask import GMM_Mask_Numba


def fill_largest_contour(motion_mask: np.ndarray, dilate_r: int = 18, erode_r: int = 6) -> np.ndarray:
    """Return a filled binary mask from the largest contour in motion_mask.

    1. Dilate to bridge MOG2 gaps
    2. Find contours, keep the largest by area
    3. Fill it solid
    4. Erode back to avoid over-bloating
    """
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_r * 2 + 1,) * 2)
    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_r  * 2 + 1,) * 2)

    dilated = cv2.dilate(motion_mask, k_d)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(motion_mask)

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(motion_mask)
    cv2.fillPoly(filled, [largest], 255)
    return cv2.erode(filled, k_e)


def soft_alpha_composite(frame: np.ndarray, bg_prob: np.ndarray,
                         blur_ksize: int = 15, blur_sigma: float = 5.0) -> np.ndarray:
    """Blend sharp foreground and blurred background using bg_prob as alpha.

    alpha = 0  (bg_prob=0, certain fg) -> sharp frame pixel
    alpha = 1  (bg_prob=1, certain bg) -> blurred pixel
    """
    frame_u8 = frame.astype(np.uint8)
    blurred  = cv2.GaussianBlur(frame_u8, (blur_ksize, blur_ksize), blur_sigma)

    alpha = bg_prob[:, :, np.newaxis]           # (H, W, 1) float32 in [0, 1]
    composite = frame_u8 * (1.0 - alpha) + blurred * alpha
    return composite.astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='0')
    args = parser.parse_args()

    input_path = 0 if args.input_path == '0' else args.input_path
    cap = cv2.VideoCapture(input_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    gmm_mask = GMM_Mask_Numba(height, width)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_f32 = np.ascontiguousarray(frame, dtype=np.float32)

        motion_mask, bg_prob, _ = gmm_mask.apply(frame_f32)

        # ── approach 1: fill largest contour ──────────────────────────────────
        fg_mask   = fill_largest_contour(motion_mask)
        blurred   = cv2.GaussianBlur(frame, (15, 15), 5.0)
        contour_result = np.where(fg_mask[:, :, np.newaxis] > 0, frame, blurred)

        # ── approach 2: soft alpha blend from bg_prob directly ────────────────
        soft_result = soft_alpha_composite(frame_f32, bg_prob)

        cv2.imshow("MOG2 Mask",       motion_mask)
        cv2.imshow("Contour Fill",    contour_result.astype(np.uint8))
        cv2.imshow("Soft Alpha",      soft_result)

        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
