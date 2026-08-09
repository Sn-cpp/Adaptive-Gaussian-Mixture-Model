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

    running = True if input_source.isOpened() else False

    while True:
        flag, first_frame = input_source.read()
        if flag:
            break


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

    model = model_list[model_choice][1](to_model(first_frame), n_components=MOG2_N_COMPONENTS, parallel=True)


    # --------------------------------------------------------------------------------------
    # Running

    # Warmup the GPU (only the CuPy model needs it)
    if model_choice == 3 and cp_gpu_warmup is not None:
        cp_gpu_warmup()

    print("Ready")
    while running:
        flag, frame = input_source.read()

        if not flag:
            running = False
            continue

        # Convert the frame to planar mode (C, H, W)
        planar_frame = to_model(frame).transpose(2, 0, 1).astype(np.float32)

        mask, time_cost = model.step(planar_frame)
        model_fps = int(1 / max(time_cost, 1e-9))

        # mask_refiner binarises: MOG2's shadow value (127) counts as background.
        refined_mask = mask_refiner(mask)
        result = background_subtractor(frame, refined_mask)

        cv2.putText(result,
            f"{model_fps}",
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
