"""Background subtraction + selective blur on traffic video.

    python main.py --input_path highway.mp4 --model numba
    python main.py --input_path highway.mp4 --model cuda_v2 --no-display

Defaults are the configuration `eval_highway.py` scored highest on CDnet
`highway` (F1 0.9843): YCrCb model input, foreground where the background
confidence is below 0.5, median 5, flood-fill the holes.

`cuda_v1` and `cuda_v2` take a different route through this file: they convert
colour, threshold, median-filter, blur and composite on the device, so the only
host work left per frame is the flood fill. Everything else keeps the original
host chain, which is the specification those kernels are tested against.
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
                    help="skip the hole fill (costs about 0.4 F1). On the GPU "
                         "backends this also keeps the mask on the device, "
                         "saving both of its transfers")
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
    # v1/v2 do the colour conversion on the device, so they take the raw BGR
    # frame and are told which space they are converting to. Everything else
    # gets a converted frame from the host.
    # Ask the class whether it takes `colorspace` rather than probing with
    # try/except TypeError: that would also swallow a genuine TypeError raised
    # from inside __init__ and quietly construct a model with the wrong
    # colour space, which is exactly the kind of silent-wrong-number bug this
    # pipeline is built to avoid.
    import inspect
    kw = ({"colorspace": args.colorspace}
          if "colorspace" in inspect.signature(cls.__init__).parameters else {})
    model = cls(height, width, **kw)
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
        t0 = time.perf_counter()
        if gpu_post:
            # The whole frame stays on the device except for the flood fill:
            # BGR up (3 bytes a pixel), mask down, filled mask back up, and the
            # composite down. The fill is the one stage that is inherently
            # sequential -- see utils/post_processing.fill_holes.
            if args.no_fill:
                model.mask_from_bgr(frame, to_host=False)
                refined = None
                result = model.composite()
            else:
                refined = fill_holes(model.mask_from_bgr(frame))
                result = model.composite(refined)
        else:
            # Hand `apply()` the uint8 frame: it calls to_planar(), which
            # transposes and casts in one pass. Casting to float32 here first
            # made that cast a second full-frame copy -- 2.2 ms a frame at
            # 1080p, buying nothing, since to_planar produces the identical
            # array either way.
            mask, bg_prob, _ = model.apply(to_model(frame))
            refined = refine_mask(np.asarray(mask), bg_prob=bg_prob,
                                  do_fill=not args.no_fill)
            result = background_blur(frame, refined, BLUR_KSIZE, BLUR_SIGMA)
        total += time.perf_counter() - t0
        frames += 1

        if not args.no_display:
            cv2.putText(result, f"{frames / max(total, 1e-9):.1f} FPS",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if refined is not None:
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
