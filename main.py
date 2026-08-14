"""Background subtraction + selective blur on traffic video.

    python main.py --input_path highway.mp4 --model numba
    python main.py --input_path highway.mp4 --model cuda_v2 --no-display

Defaults are the configuration `eval_highway.py` scored highest on CDnet
`highway` (F1 0.9843): YCrCb model input, foreground where the background
confidence is below 0.5, median 5, flood-fill the holes.
"""
from gmm_mask import warmup_mask_gmm_jit
warmup_mask_gmm_jit()

import argparse
import time

import cv2
import numpy as np

from gmm_mask import (GMM_Mask_CPU, GMM_Mask_CUDA, GMM_Mask_CUDA_v1,
                      GMM_Mask_CUDA_v2, GMM_Mask_CuPy, GMM_Mask_Numba)
from settings import BLUR_KSIZE, BLUR_SIGMA
from utils.post_processing import background_blur, fill_holes, refine_mask

MODELS = {
    "cpu": GMM_Mask_CPU,
    "numba": GMM_Mask_Numba,
    "cuda": GMM_Mask_CUDA,
    "cuda_v1": GMM_Mask_CUDA_v1,
    "cuda_v2": GMM_Mask_CUDA_v2,
    "cupy": GMM_Mask_CuPy,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_path", default="0")
    ap.add_argument("--model", default="numba", choices=sorted(MODELS))
    ap.add_argument("--colorspace", default="ycrcb", choices=("bgr", "ycrcb"),
                    help="what the model sees; the composite is always BGR. "
                         "ycrcb is worth +16 F1 on highway (0.827 -> 0.984)")
    ap.add_argument("--no-fill", action="store_true",
                    help="skip the hole fill (costs about 0.4 F1)")
    ap.add_argument("--no-display", action="store_true",
                    help="for headless runs; still prints throughput")
    args = ap.parse_args()

    cls = MODELS[args.model]
    if cls is None:
        raise SystemExit(
            f"--model {args.model} is unavailable: its GPU dependency "
            "(cupy or numba.cuda) is not installed on this machine.")

    src = 0 if args.input_path == "0" else args.input_path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input_path!r}")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    to_model = ((lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb))
                if args.colorspace == "ycrcb" else (lambda f: f))
    model = cls(height, width)
    # The CUDA v1/v2 classes run threshold+median on the device and hand back a
    # mask that only needs the flood fill; everything else returns MOG2's raw
    # mask plus the confidence map, and the host does the rest.
    gpu_post = getattr(model, "post", False)

    print(f"{args.model} | {width}x{height} | {args.colorspace} | "
          f"post on {'GPU' if gpu_post else 'host'}")

    frames, total = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        planar_in = np.ascontiguousarray(to_model(frame), dtype=np.float32)

        t0 = time.perf_counter()
        mask, bg_prob, _ = model.apply(planar_in)
        if gpu_post:
            # threshold and median already ran on the device; only the fill left
            refined = np.asarray(mask)
            if not args.no_fill:
                refined = fill_holes(refined)
        else:
            refined = refine_mask(np.asarray(mask), bg_prob=bg_prob,
                                  do_fill=not args.no_fill)
        result = background_blur(frame, refined, BLUR_KSIZE, BLUR_SIGMA)
        total += time.perf_counter() - t0
        frames += 1

        if not args.no_display:
            cv2.putText(result, f"{frames / max(total, 1e-9):.1f} FPS",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("mask", refined)
            cv2.imshow("composite", result)
            if cv2.waitKey(1) in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    if frames:
        print(f"{frames} frames, {total / frames * 1000:.2f} ms/frame, "
              f"{frames / total:.1f} FPS")


if __name__ == "__main__":
    main()
