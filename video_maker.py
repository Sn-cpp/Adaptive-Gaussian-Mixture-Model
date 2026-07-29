import cv2
import os
import glob

def images_to_video(image_folder, output_video_path, file_type: str="jpg", output_video_len_in_seconds: int=-1, fps=30):
    # Find all images in the directory
    # Using glob ensures we can match extensions easily
    image_files = glob.glob(os.path.join(image_folder, f"*.{file_type}"))
    
    # Sort files alphanumeric so they sequence correctly (e.g., frame1, frame2...)
    image_files.sort(key=lambda f: [int(c) if c.isdigit() else c for c in os.path.split(f)[-1].replace('.', '').split()])

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

# Example Usage:
images_to_video('datasets/dataset2012/dataset/baseline/highway/input', 'input.mp4', 'jpg', 120, fps=30)
images_to_video('datasets/dataset2012/dataset/baseline/highway/groundtruth', 'groundtruth.mp4', 'png', 120, fps=30)
