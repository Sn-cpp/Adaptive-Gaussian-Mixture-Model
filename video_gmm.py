import argparse

import cv2
import numpy as np
from tqdm import tqdm

from gmm.cpu.GMM_cpu import GMM_CPU


def gmm_video(input_path: str, output_path: str, fps: float,
              n_components: int = 10,
              match_threshold: np.float32 = np.float32(3.5),
              update_alpha: np.float32 = np.float32(0.01),
              weight_threshold: np.float32 = np.float32(0.7),
              fraction: float = 0.25):
    # Initialize the video capture object
    cap = cv2.VideoCapture(input_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Add the Mat frame to our list
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {input_path}")

    height, width, layers = frames[0].shape
    video_dims = (width, height)

    # Define the codec for MP4 and initialize VideoWriter
    # 'mp4v' is universally supported for .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_dims)

    process_frames = frames[:max(1, int(len(frames) * fraction))]
    model = GMM_CPU(process_frames[0], n_components=n_components)

    for frame in tqdm(process_frames):
        # The models take planar (C, H, W) float32 frames, and step() returns
        # (mask, seconds).
        planar_frame = frame.transpose(2, 0, 1).astype(np.float32)
        mask, _ = model.step(planar_frame, update_alpha)

        video_writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the GMM mask of a video to a file")
    parser.add_argument("--input_path", type=str, default="input.mp4")
    parser.add_argument("--output_path", type=str, default="masks_quarter.mp4")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--n_components", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=0.25,
                        help="Fraction of the clip to process")
    args = parser.parse_args()

    gmm_video(args.input_path, args.output_path, args.fps,
              n_components=args.n_components, fraction=args.fraction)
