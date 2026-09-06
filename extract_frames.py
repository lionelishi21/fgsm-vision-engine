import cv2
import os
import argparse

def extract_frames(video_path, output_dir, frame_interval=6):
    """
    Extracts frames from a video file and saves them to an output directory.
    
    :param video_path: Path to the input video file.
    :param output_dir: Directory where the extracted frames will be saved.
    :param frame_interval: Extract one frame every `frame_interval` frames. 
                           For a 60 FPS video, setting this to 6 gives 10 FPS output.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get some video properties for logging
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opened video: {video_path}")
    print(f"FPS: {fps}, Total Frames: {total_frames}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        
        # Break the loop if there are no more frames
        if not ret:
            break
            
        # Save the frame if it matches the interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Done! Extracted {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a fighting game video for dataset generation.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (.mp4, etc.)")
    parser.add_argument("--out", type=str, default="output", help="Directory to save the extracted frames")
    parser.add_argument("--interval", type=int, default=6, help="Extract every Nth frame (default: 6)")
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.out, args.interval)
