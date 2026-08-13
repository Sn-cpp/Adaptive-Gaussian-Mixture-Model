import cv2
import argparse
import numpy as np
from time import sleep
import keyboard as kb

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
        3: ("CuPy RawKernel", GMM_CUPY)
    }


    # --------------------------------------------------------------------------------------
    # Arguments handling
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, help="Input source")
    parser.add_argument("--groundtruth_path", type=str, help="Groundtruth source")
    parser.add_argument("--model", type=int, default=0, help="""
        Model selection: """ + " | ".join([f"{idx} - {name}" for idx, (name, _) in model_list.items()]))
    parser.add_argument("--n_components", type=int, default=7, help="Number of Gaussian components per pixel")
    parser.add_argument("--update_alpha", type=float, default=0.01, help="Updating rate for components")
    parser.add_argument("--match_threshold", type=float, default=3.5, help="Background matching threshold")
    parser.add_argument("--weight_threshold", type=float, default=0.7, help="Cumulative weight threshold for components")

    args = parser.parse_args()

    input_path = args.input_path
    groundtruth_path = args.groundtruth_path

    if not (0 <= args.model < len(model_list.keys())):
        raise ValueError("Unknown model")

    if not (1 <= args.n_components <= MAX_COMPONENTS):
        raise ValueError("Invalid number of Gaussian components")


    # --------------------------------------------------------------------------------------
    # Input initialization

    input_source = cv2.VideoCapture(input_path)
    groundtruth_source = cv2.VideoCapture(groundtruth_path)

    CAM_WIDTH = int(input_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAM_HEIGHT = int(input_source.get(cv2.CAP_PROP_FRAME_HEIGHT))

    running = True if input_source.isOpened() else False

    if not input_source.isOpened():
        raise SystemExit(f"Cannot open input source {input_path!r}.")
    if not groundtruth_source.isOpened():
        raise SystemExit(f"Cannot open groundtruth source {groundtruth_path!r}.")

    for _ in range(30):
        input_flag, first_frame = input_source.read()
        gth_flag, _ = groundtruth_source.read()
        if input_flag and gth_flag:
            break
    else:
        raise SystemExit("Input or groundtruth opened but produced no frame in 30 tries.")
    

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
    cv2_gmm = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # CPU baseline
    base_model = GMM_CPU(first_frame, n_components=gaussian_components)

    # User-chosen model
    model = model_list[model_choice][1](first_frame, n_components=gaussian_components, parallel=True)


    # --------------------------------------------------------------------------------------
    # Utilities 

    # Execution time
    cpu_fps_last = 0
    model_fps_last = 0
    fps_graph = Monitor(800, 400, 360, np.arange(30, 360, 30), "FPS", f"Green-CPU base | Red-{model_list[model_choice][0]}", (5, 15))

    # Intersection over Union (IOU) metric
    iou_mog2_last = 0
    iou_cpu_last = 0
    iou_model_last = 0
    iou_graph = Monitor(400, 400, 1.0, np.round(np.arange(0.1, 1.0, 0.1), 2), "IOU", f"Blue-MOG2 | Green-CPU base | Red-{model_list[model_choice][0]}", (5, 15))

    # Masks-difference in Mean Absolute Error (MAE)
    mask_mae_last = 0
    mae_graph = Monitor(400, 400, 0.01, np.round(np.arange(0.001, 0.01, 0.001), 3), "CPU vs Model Masks MAE")

    # Number of different pixels
    diff_max = (CAM_WIDTH * CAM_HEIGHT) // 8
    mask_diff_last = 0
    mask_diff_graph = Monitor(400, 400, diff_max, np.arange(1000, diff_max, 1000), "CPU vs Model Different pixels count")

    # --------------------------------------------------------------------------------------
    # Running

    # Warmup the GPU (only the CuPy model needs it)
    if model_choice == 3 and cp_gpu_warmup is not None:
        cp_gpu_warmup()

    print("Sanity check:")
    print("cv2_gmm:    ", type(cv2_gmm))
    print("base_model: ", type(base_model))
    print("model:      ", type(model))
    print("Ready")

    pause = False
    while running:
        if kb.is_pressed("space"):
            pause = not pause
            sleep(0.1)
            print(pause)
    
        if not pause:
            input_flag, frame = input_source.read()
            gth_flag, ground = groundtruth_source.read()

            if not (input_flag and gth_flag):
                running = False
                continue

            # Convert the frame to planar mode (C, H, W) with C=3 (BGR)
            planar_frame = frame.transpose(2, 0, 1).astype(np.float32)
            ground = cv2.cvtColor(ground, cv2.COLOR_BGR2GRAY)
            _, ground = cv2.threshold(ground, 127, 255, cv2.THRESH_BINARY)

            # Predict mask
            mask_mog2 = cv2_gmm.apply(frame)

            mask_cpu, cpu_cost = base_model.step(planar_frame, update_alpha)

            mask_model, model_cost = model.step(planar_frame, update_alpha)

            # Post-process masks
            refined_mask_mog2 = mask_refiner(mask_mog2)
            # refined_cv2_gmm = cv2.threshold(refined_cv2_gmm, 127, 255, cv2.THRESH_BINARY) # MOG2 somehow produces gray region

            refined_cpu = mask_refiner(mask_cpu)
            refined_model = mask_refiner(mask_model)

            # Combine groundtruth and masks into a single frame
            mask_grid = cv2.vconcat([
                np.full(shape=(30, 2*CAM_WIDTH + 3), fill_value=127, dtype=np.uint8),
                cv2.hconcat([ground, np.full(shape=(CAM_HEIGHT, 3), fill_value=127, dtype=np.uint8), refined_mask_mog2]),
                np.full(shape=(30, 2*CAM_WIDTH + 3), fill_value=127, dtype=np.uint8),
                cv2.hconcat([refined_cpu, np.full(shape=(CAM_HEIGHT, 3), fill_value=127, dtype=np.uint8), refined_model])
            ])

            # Put description
            cv2.putText(mask_grid, "Groundtruth", (5, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, 255, 1)
            cv2.putText(mask_grid, "opencv's MOG2", (CAM_WIDTH + 8, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, 255, 1)
            cv2.putText(mask_grid, "CPU Baseline", (5, 50 + CAM_HEIGHT), cv2.FONT_HERSHEY_TRIPLEX, 0.7, 255, 1)
            cv2.putText(mask_grid, f"{model_list[model_choice][0]}", (CAM_WIDTH + 8, 50 + CAM_HEIGHT), cv2.FONT_HERSHEY_TRIPLEX, 0.7, 255, 1)

            cv2.imshow("Groundtruth and Models' masks", mask_grid)

            # Write FPS values
            fps_cpu = 1.0 / max(cpu_cost, 1e-9)
            fps_model = 1.0 / max(model_cost, 1e-9)
            fps_graph.write_value(fps_cpu, cpu_fps_last, (0, 255, 0))
            fps_graph.write_value(fps_model, model_fps_last, (0, 0, 255))
            fps_graph.display(f"CPU: {fps_cpu:.2f} | {model_list[model_choice][0]}: {fps_model:.1f}", (5, 30))

            # Compute IOU for each models
            iou_mog2 = compute_iou(refined_mask_mog2, ground)
            iou_cpu = compute_iou(refined_cpu, ground)
            iou_model = compute_iou(refined_model, ground)

            # Write latest value to graph for displaying
            iou_graph.write_value(iou_mog2, iou_mog2_last, (255, 0, 0))
            iou_graph.write_value(iou_cpu, iou_cpu_last, (0, 255, 0))
            iou_graph.write_value(iou_model, iou_model_last, (0, 0, 255))
            iou_graph.display()

            # Compute MAE between CPU and Model masks
            # Both masks are uint8, so a plain subtraction wraps before np.abs
            # sees it: cpu=0 against model=255 gave 1, not 255, understating
            # the disagreement 255-fold. They are binary anyway.
            abs_err = refined_cpu != refined_model
            mae = abs_err.mean()
            mae_graph.write_value(mae, mask_mae_last, (0, 0, 255))
            mae_graph.display(f"{mae:2f}", (300, 15))

            # Count number of different pixels
            diff_count = np.count_nonzero(abs_err)
            mask_diff_graph.write_value(diff_count, mask_diff_last, (0, 0, 255))
            mask_diff_graph.display(f"Diff count: {diff_count}", (300, 15))

            # Save to histories
            cpu_fps_last = fps_cpu
            model_fps_last = fps_model
            iou_mog2_last = iou_mog2
            iou_cpu_last = iou_cpu
            iou_model_last = iou_model
            mask_mae_last = mae
            mask_diff_last = diff_count

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            running = False
