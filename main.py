from gmm_mask import warmup_mask_gmm_jit
warmup_mask_gmm_jit()

import argparse
import cv2
import numpy as np

from gmm_mask import GMM_Mask_Numba

_BG_PROB_THRESH = np.float32(0.65)


def _ellipse(r: int):
    s = r * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))


def make_morph_kernels(height: int, width: int):
    short    = min(height, width)
    close_r  = max(3, int(short * 0.019))
    dilate_r = max(5, int(short * 0.042))
    erode_r  = max(3, int(short * 0.027))
    return _ellipse(close_r), _ellipse(dilate_r), _ellipse(erode_r)


def connect_foreground(mask: np.ndarray,
                       k_close, k_dilate, k_erode) -> np.ndarray:
    expanded = cv2.dilate(mask, k_dilate)

    closed = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    filled = np.zeros_like(mask)
    cv2.fillPoly(filled, contours, 255)

    result = cv2.erode(filled, k_erode)

    # n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    # if n_labels <= 1:
    #     return result
    # largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    # lr_comp = np.where(labels == largest, np.uint8(255), np.uint8(0))

    img = cv2.bitwise_not(result)

    h, w = img.shape[:2]

    # Mask size must be (h + 2, w + 2)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Starting position
    seed_point = (50, 50)
    new_color = (0, 255, 0) # Green color in BGR

    # Threshold bounds
    lo_diff = (10, 10, 10)
    up_diff = (10, 10, 10)

    # Apply floodFill
    cv2.floodFill(img, mask, seed_point, new_color, lo_diff, up_diff, flags=4 | cv2.FLOODFILL_FIXED_RANGE)

    ret = cv2.bitwise_or(result, img)

    ret = cv2.erode(ret, k_erode)
    
    return ret


from numba import njit, prange

@njit(parallel=True, cache=True)
def foo(b_prob: np.ndarray, sobel_mask: np.ndarray):
    H, W = b_prob.shape

    out = np.zeros_like(b_prob, dtype=np.uint8)

    for i in prange(H):
        for j in range(W):
            if sobel_mask[i, j] < 10:
                out[i, j] = 0
            elif b_prob[i, j] > 0.2:
                out[i, j] = 0
            else:
                out[i, j] = 255

    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='0')
    args = parser.parse_args()

    input_path = 0 if args.input_path == '0' else args.input_path
    cap = cv2.VideoCapture(input_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    k_close, k_dilate, k_erode = make_morph_kernels(height, width)

    gmm = GMM_Mask_Numba(height, width)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_f32 = np.ascontiguousarray(frame, dtype=np.float32)

        gray = cv2.cvtColor(frame_f32, cv2.COLOR_BGR2GRAY)

        # Tính đạo hàm theo hướng X (cạnh đứng)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

        # Tính đạo hàm theo hướng Y (cạnh ngang)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Chuyển đổi về lại kiểu uint8 (0-255) để hiển thị
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)

        # Kết hợp cả hai hướng (độ lớn gradient)
        combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

        # cv2.imshow("Sobel", combined)

        motion_mask, bg_prob, _ = gmm.apply(frame_f32)

        prob_u8 = (np.clip(bg_prob, 0.0, 1.0) * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
        cv2.imshow("BG/FG Probabilities", heatmap)

        foo_res = foo(bg_prob, combined)

        med_mask = cv2.medianBlur(foo_res, 5)

        clean_mask = connect_foreground(med_mask, k_close, k_dilate, k_erode)

        cv2.imshow("Filtered Combined Mask", med_mask)
        cv2.imshow("Post-processed Mask", clean_mask)

        blur = cv2.GaussianBlur(frame, (15, 15), 5.0)

        fg = np.zeros_like(frame, dtype=np.uint8)
        
        cv2.copyTo(frame, clean_mask, blur)
        cv2.copyTo(frame, clean_mask, fg)

        cv2.imshow("Foreground Cut", fg)
        cv2.imshow("Final Composite", blur)



        # fg_mask   = connect_foreground(motion_mask, bg_prob, k_close, k_dilate, k_erode)
        # blurred   = cv2.GaussianBlur(frame, (15, 15), 5.0)
        # composite = np.where(fg_mask[:, :, np.newaxis] > 0, frame, blurred)

        # cv2.imshow("GMM Mask",     motion_mask)
        # cv2.imshow("Connected FG", fg_mask)
        # cv2.imshow("Composite",    composite.astype(np.uint8))

        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
