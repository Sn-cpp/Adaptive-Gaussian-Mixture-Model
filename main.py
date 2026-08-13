import cv2
import argparse
import numpy as np

from settings import *
from utils import *
from gmm import *

if __name__ == "__main__":
    # --------------------------------------------------------------------------------------
    # Models declaration

    model_list = {
        0: ("CPU", GMM_CPU),
        1: ("Numba", GMM_CPU_NUMBA),
        2: ("CUDA", GMM_CUDA),
        3: ("CuPy RawKernel", GMM_CUPY),
    }

    # Every model reproduces OpenCV's BackgroundSubtractorMOG2, with its
    # calibrated thresholds in settings.MOG2_*. A negative update_alpha selects
    # OpenCV's warm-up ramp 1/min(2*nframes, history), which is the default.


    # --------------------------------------------------------------------------------------
    # Arguments handling
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="0", help="Input source")
    parser.add_argument("--model", type=int, default=1, help="""
        Model selection: """ + " | ".join([f"{idx} - {name}" for idx, (name, _) in model_list.items()]))
    parser.add_argument("--colorspace", type=str, default="bgr", choices=("bgr", "ycrcb"),
                        help="Colorspace the model sees. ycrcb separates luma from chroma "
                             "(+6 to +13 F1 on CDnet highway); display stays BGR either way")
    parser.add_argument("--conservative", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze the background model wherever the previous frame was "
                             "foreground (ViBe's rule), so a subject who stops moving is not "
                             "absorbed. On by default here; --no-conservative gives plain "
                             "MOG2, which is what the cv2 parity tests use")
    parser.add_argument("--clean-plate", type=int, default=0, metavar="N",
                        help="Train the background model on the first N frames before "
                             "showing anything, and step out of shot while it does. MOG2 "
                             "cannot tell a person who was there from frame 0 apart from a "
                             "wall; N empty frames are what makes that distinction possible. "
                             "Try 48 (2s at 24fps). 0 disables it")

    args = parser.parse_args()

    input_path = 0 if args.input_path == "0" else args.input_path

    if not (0 <= args.model < len(model_list.keys())):
        raise ValueError("Unknown model")

    if model_list[args.model][1] is None:
        raise ValueError(f"Model {args.model} ({model_list[args.model][0]}) is unavailable — "
                         "its GPU dependency (cupy or numba.cuda) is not installed")


    # --------------------------------------------------------------------------------------
    # Input initialization

    input_source = cv2.VideoCapture(input_path)

    CAM_WIDTH = input_source.get(cv2.CAP_PROP_FRAME_WIDTH)
    CAM_HEIGHT = input_source.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if not input_source.isOpened():
        raise SystemExit(f"Cannot open input source {input_path!r}. Pass a video "
                         f"file with --input_path, or 0 for the default camera.")

    running = True

    # A capture that opens but yields nothing (a disconnected camera, a codec the
    # build cannot decode) used to spin here for ever with no output at all.
    for _ in range(30):
        flag, first_frame = input_source.read()
        if flag:
            break
    else:
        raise SystemExit(f"{input_path!r} opened but produced no frame in 30 tries.")


    # --------------------------------------------------------------------------------------
    # Model initialization

    # Model selection
    model_choice = args.model

    # The model can watch a different colorspace than the one we display: the
    # mask is colorspace-agnostic, and the blur composite always uses the BGR
    # frame. YCrCb separates luma from chroma, which measurably cleans up the
    # mask under shadows.
    use_ycrcb = args.colorspace == "ycrcb"

    to_model = (lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)) if use_ycrcb else (lambda f: f)

    def to_planar(f):
        return to_model(f).transpose(2, 0, 1).astype(np.float32)

    model = model_list[model_choice][1](to_model(first_frame), n_components=MOG2_N_COMPONENTS,
                                        conservative=args.conservative, parallel=True)


    # --------------------------------------------------------------------------------------
    # Running

    # Warmup the GPU (only the CuPy model needs it)
    if model_choice == 3 and cp_gpu_warmup is not None:
        cp_gpu_warmup()

    # Clean plate. Nothing here is special-cased in the model: these are
    # ordinary steps, they just happen to see a scene with nobody in it, so
    # mode 0 ends up holding the real background instead of the background
    # plus whoever was sitting in front of the camera when it started.
    # Conservative update is what then keeps that model intact.
    if args.clean_plate > 0:
        print(f"Clean plate: step out of shot, learning the empty scene for "
              f"{args.clean_plate} frames...")
        for _ in range(args.clean_plate):
            flag, frame = input_source.read()
            if not flag:
                print("  input ended during clean-plate capture")
                break
            model.step(to_planar(frame))
        print("Clean plate done — come back into shot.")

    # A backend that leaves bg_prob at zero would segment a field of zeros, so
    # ask rather than assume — see MOG2Base.FILLS_BG_PROB.
    use_bg_prob = type(model).FILLS_BG_PROB
    close_k = close_ksize_for(first_frame)
    print(f"Post-processing: {'bg_prob' if use_bg_prob else 'binary mask'} "
          f"+ median 5 + CLOSE {close_k} + hole fill")

    print("Ready")
    while running:
        flag, frame = input_source.read()

        if not flag:
            running = False
            continue

        mask, time_cost = model.step(to_planar(frame))
        # after step, not before: GMM_CUPY rebinds self.bg_prob to a fresh host
        # array every frame rather than filling one in place, so a reference
        # captured earlier would be the previous frame's.
        bg_prob = model.bg_prob if use_bg_prob else None
        # Not int(): the sequential reference runs below 1 FPS, and the whole
        # point of the comparison is that number. int() rendered it as "0".
        model_fps = 1.0 / max(time_cost, 1e-9)

        # mask_refiner binarises: MOG2's shadow value (127) counts as background.
        # bg_prob is a better foreground decision than the binary mask (+3.0 F1
        # on highway) and every backend fills it; the CLOSE bridges the gaps a
        # person's low-texture interior leaves. Both are measured in its
        # docstring.
        refined_mask = mask_refiner(mask, bg_prob=bg_prob, close_ksize=close_k)
        result = background_subtractor(frame, refined_mask)

        cv2.putText(result,
            f"{model_fps:.2f} FPS" if model_fps < 10 else f"{model_fps:.0f} FPS",
            (5, 25),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 255, 0),
            1)

        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            running = False
