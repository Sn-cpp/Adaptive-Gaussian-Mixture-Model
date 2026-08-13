from gmm_em import warmup_em_gmm_jit
from gmm_mask import warmup_mask_gmm_jit
from grabcut import warmup_grabcut_jit
warmup_em_gmm_jit()
warmup_mask_gmm_jit()
warmup_grabcut_jit()

import argparse
import cv2
import numpy as np

from gmm_mask import GMM_Mask_Numba
from grabcut import GrabCut_CUDA_v0, GrabCut_CUDA_v1, GrabCut_CUDA_v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='0')
    args = parser.parse_args()

    input_path = 0 if args.input_path == '0' else args.input_pathq
    cap = cv2.VideoCapture(0)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    gmm_mask = GMM_Mask_Numba(height, width)
    grabcut  = GrabCut_CUDA_v2(height, width)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.ascontiguousarray(frame, dtype=np.float32)

        motion_mask, bg_prob, _ = gmm_mask.apply(frame, to_host=False)
        mask, result, elapsed   = grabcut.apply(frame, bg_prob)

        # cv2.imshow("GMM Mask",    motion_mask)
        cv2.imshow("Final Mask",  mask)
        cv2.imshow("Result",      result)

        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
