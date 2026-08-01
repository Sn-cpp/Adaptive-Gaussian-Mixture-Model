import cv2
import argparse
import numpy as np
from time import perf_counter

from settings import *
from utils import *
from gmm import *

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
    parser.add_argument("--update_alpha", type=float, default=0.1, help="Updating rate for components")
    parser.add_argument("--match_threshold", type=float, default=3, help="Background matching threshold")
    parser.add_argument("--weight_threshold", type=float, default=0.9, help="Cumulative weight threshold for components")

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

    
    # cv2's GMM
    cv2_gmm = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=match_threshold, detectShadows=False)

    model = model_list[model_choice][1](first_frame, n_components=gaussian_components, parallel=True)
    

    # --------------------------------------------------------------------------------------
    # Utilities 
    model_fps_graph = Monitor(400, 400, 150, np.arange(30, 150, 30), f"{model_list[model_choice][0]} FPS Monitor")
    last_fps = 0


    # --------------------------------------------------------------------------------------
    # Running

    # Warmup the GPU (if used)
    if model_choice >= 2:
        cp_gpu_warmup()

    print(cv2_gmm.getHistory())
    print(cv2_gmm.getBackgroundRatio())
    print(cv2_gmm.getNMixtures())
    print(cv2_gmm.getVarInit())
    print(cv2_gmm.getVarThresholdGen())
    print(cv2_gmm.getHistory())

    print("Ready")
    i = 0
    while running:
        flag, frame = input_source.read()

        if not flag:
            running = False
            continue
        i += 1

        mog2_mask = cv2_gmm.apply(frame)

        # Convert the frame to planar mode (C, H, W) with C=3 (BGR)
        planar_frame = frame.transpose(2, 0, 1).astype(np.float32)

        mask, time_cost = model.step(planar_frame, match_threshold, update_alpha, weight_threshold, 9.0)
        model_fps = int(1/time_cost)

        # _, write_cost = cpu_timer(model_fps_graph.write_value, value=model_fps, last_value=last_fps, color=(0, 0, 255))
        # print(write_cost)

        # refined_mask = mask_refiner(mask)
        # result = background_subtractor(frame, refined_mask)
        # refined_mask, refine_cost = cpu_timer(mask_refiner, mask=mask)
        # result, subtract_cost = cpu_timer(background_subtractor, frame=frame, mask=refined_mask)
        # print("Refine cost: ", refine_cost)
        # print("Subtraction cost: ", subtract_cost)


        cv2.imshow("Raw Mask", mask)
        cv2.imshow("MOG2 Mask", mog2_mask)
        # cv2.imshow("Result", result)

        # start = perf_counter()
        # model_fps_graph.write_value(model_fps, last_fps, (0, 0, 255))
        # last_fps = model_fps
        # model_fps_graph.display(f"{model_fps}", (5, 15))
        # end = perf_counter()
        # print(end - start)



        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            running = False
