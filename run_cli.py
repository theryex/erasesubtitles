import os
import argparse
import time
from video_processing import erase_subtitles

def main_cli():
    parser = argparse.ArgumentParser(
        description="A command-line tool to remove subtitles from a video file."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="The path to the video file (e.g., 'my_movie.mp4' or '/path/to/my_movie.mp4')."
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="The ID of the GPU to use for processing (e.g., 0, 1)."
    )
    args = parser.parse_args()

    print("Ensuring required directories exist...")
    os.makedirs("Input/Video", exist_ok=True)
    os.makedirs("Input/Audio", exist_ok=True)
    os.makedirs("Output", exist_ok=True)

    video_path = args.video

    # If the path doesn't exist, assume it's in the Input/Video directory
    if not os.path.exists(video_path):
        input_video_path = os.path.join("Input/Video", video_path)
    else:
        input_video_path = video_path

    if not os.path.exists(input_video_path):
        print(f"\nERROR: Input file not found at '{input_video_path}'")
        print("Please make sure your video is placed in the 'Input/Video' folder or provide a full path.")
        return  # Exit the script

    video_filename = os.path.basename(input_video_path)

    print(f"\nStarting subtitle removal for '{video_filename}'...")
    start_time = time.time()

    output_video_path, message = erase_subtitles(input_video_path, gpu_id=args.gpu_id)

    end_time = time.time()
    duration = end_time - start_time

    if output_video_path:
        print("\n--- Process Complete ---")
        print(f"✅ Success: {message}")
        print(f"Output file saved to: {output_video_path}")
        print(f"Total processing time: {duration:.2f} seconds")
    else:
        print("\n--- Process Failed ---")
        print(f"❌ Error: {message}")

if __name__ == "__main__":
    main_cli()