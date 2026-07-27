import cv2
import numpy as np
from gmm.cpu.GMM_cpu import GMM_CPU
from tqdm import tqdm

def gmm_video(input_path: str, output_path: str, fps: float):
    # Initialize the video capture object
    cap = cv2.VideoCapture(input_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Add the Mat frame to our list
        frames.append(frame)

    height, width, layers = frames[0].shape
    video_dims = (width, height)

    # Define the codec for MP4 and initialize VideoWriter
    # 'mp4v' is universally supported for .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_dims)

    process_frames = frames[:len(frames) // 4]
    model = GMM_CPU(process_frames[0], n_components=10)

    for idx, frame in enumerate(tqdm(process_frames)):
        mask = model.step(frame)

        video_writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    
    video_writer.release()
    cap.release()

gmm_video('output_sequence.mp4', 'masks_quarter.mp4', 30)