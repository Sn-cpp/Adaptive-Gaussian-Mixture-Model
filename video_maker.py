import argparse
import cv2
import os
import glob
import re

def images_to_video(image_folder, output_video_path, file_type: str="jpg", output_video_len_in_seconds: int=-1, fps=30):
    # Find all images in the directory
    # Using glob ensures we can match extensions easily
    image_files = glob.glob(os.path.join(image_folder, f"*.{file_type}"))
    
    # Sort files alphanumeric so they sequence correctly (e.g., frame1, frame2...)
    # str.split() with no argument splits on whitespace, and image filenames have
    # none — so the comprehension saw one long token and never isolated a digit,
    # sorting frame10 before frame1. Split on runs of digits instead.
    image_files.sort(key=lambda f: [int(c) if c.isdigit() else c
                                    for c in re.split(r'(\d+)', os.path.basename(f))])

    if not image_files:
        print(f"No .{file_type} images found in the specified directory.")
        return
    
    output_video_len_in_seconds = len(image_files) if output_video_len_in_seconds == -1 else output_video_len_in_seconds

    # Read the first image to dynamically obtain dimensions (width, height)
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    video_dims = (width, height)

    # Define the codec for MP4 and initialize VideoWriter
    # 'mp4v' is universally supported for .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, video_dims)
    num_frame = min(fps * output_video_len_in_seconds, len(image_files))

    print(f"Starting video compilation of {num_frame} frames...")
    for idx, file_path in enumerate(image_files):
        if idx == num_frame:
            break

        img = cv2.imread(file_path)
        
        # Critical Check: Resize image if it does not match the first frame's size.
        # OpenCV VideoWriter will silently drop frames that mismatch dimensions.
        if (img.shape[1], img.shape[0]) != video_dims:
            img = cv2.resize(img, video_dims)
            
        video_writer.write(img)

    # Release the video writer to finalize and save the file
    video_writer.release()
    print(f"Video successfully saved to: {output_video_path}")

if __name__ == "__main__":
    # These used to sit at module scope, so importing this file rebuilt
    # input.mp4 and groundtruth.mp4 in the repo root from a dataset path that
    # is not checked in — overwriting both with nothing.
    ap = argparse.ArgumentParser(description="Turn a folder of frames into a video.")
    ap.add_argument("image_folder")
    ap.add_argument("output_video_path")
    ap.add_argument("--file_type", default="jpg")
    ap.add_argument("--seconds", type=int, default=-1,
                    help="-1 uses every frame in the folder")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    images_to_video(a.image_folder, a.output_video_path, a.file_type, a.seconds, a.fps)
