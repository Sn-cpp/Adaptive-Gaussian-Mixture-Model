import cv2
import argparse
import numpy as np

from settings import *
from utils import *
from gmm import *

from tester import compare_diff_square_sum

if __name__ == "__main__":
    # --------------------------------------------------------------------------------------
    # Models declaration

    model_list = {
        0: ("CPU", GMM_CPU),
        1: ("Numba", GMM_CPU_NUMBA),
        2: ("CuPy vectorized", GMM_CUPY_V0),
        3: ("CuPy RawKernel", GMM_CUPY_V1)
    }


    # --------------------------------------------------------------------------------------
    # Arguments handling
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="0", help="Input source")
    parser.add_argument("--model", type=int, default=0, help="""
        Model selection: """ + " | ".join([f"{idx} - {name}" for idx, (name, _) in model_list.items()]))
    parser.add_argument("--n_components", type=int, default=7, help="Number of Gaussian components per pixel")
    parser.add_argument("--update_alpha", type=float, default=0.01, help="Updating rate for components")
    parser.add_argument("--match_threshold", type=float, default=3.5, help="Background matching threshold")
    parser.add_argument("--weight_threshold", type=float, default=0.7, help="Cumulative weight threshold for components")

    args = parser.parse_args()

    input_path = 0 if args.input_path == "0" else args.input_path

    if not (0 <= args.model < len(model_list.keys())):
        raise ValueError("Unknown model")

    if not (1 <= args.n_components <= MAX_COMPONENTS):
        raise ValueError("Invalid number of Gaussian components")


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

    # Parameters config
    gaussian_components = args.n_components
    update_alpha = np.float32(args.update_alpha)
    match_threshold = np.float32(args.match_threshold)
    weight_threshold = np.float32(args.weight_threshold)

    model = model_list[model_choice][1](first_frame, n_components=gaussian_components, parallel=True)


    # --------------------------------------------------------------------------------------
    # Utilities 

    if model_choice >= 2:
        step_func = gpu_step
        cp_gpu_warmup()
    else:
        step_func = cpu_step

    # --------------------------------------------------------------------------------------
    # Running

    print("Ready")
    while running:
        flag, frame = input_source.read()

        if not flag:
            running = False
            continue

        # Convert the frame to planar mode (C, H, W) with C=3 (BGR)
        planar_frame = frame.transpose(2, 0, 1).astype(np.float32)

        mask, time_cost = step_func(model, planar_frame, match_threshold, update_alpha, weight_threshold)
        model_fps = int(1/time_cost)

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
