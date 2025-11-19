import os
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
import math
import numpy as np
from tqdm import tqdm

from preprocessing import gen_image_frames, seg_imgs, extract_audio
from detectText import get_coords
from splitRegion import gen_regions
from inpaint import gen_frames_and_masks, inpaint_main, merge

CHUNK_SIZE = 100  # Process 100 frames at a time

def get_video_details(video_path):
    """Gets video details: total frames, fps, and dimensions."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, (0, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total_frames, fps, (width, height)

def convert_to_mp4(video_path):
    """Converts a video to MP4 format if it's not already."""
    name, ext = os.path.splitext(video_path)
    if ext.lower() == '.mp4':
        return video_path, False  # No conversion needed

    output_path = f"{name}.mp4"
    print(f"Converting {video_path} to {output_path}...")
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        return output_path, True
    except Exception as e:
        print(f"Error converting video: {e}")
        return None, False

def detect_all_subtitle_regions(video_path, total_frames, gpu_id=0):
    """
    Pass 1: Detect all subtitle regions throughout the entire video.
    Returns a dictionary mapping frame numbers to subtitle coordinates.
    """
    print("\nPass 1: Detecting all subtitle regions...")
    cap = cv2.VideoCapture(video_path)
    subtitle_map = {}

    for frame_num in tqdm(range(total_frames), desc="Detecting subtitles"):
        ret, frame = cap.read()
        if not ret:
            break

        # Use a simplified mask generation for speed
        mask = seg_imgs([frame])
        coords = get_coords(1, mask, gpu_id=gpu_id)

        if coords:
            subtitle_map[frame_num] = coords

    cap.release()
    print(f"Found subtitles in {len(subtitle_map)} frames.")
    return subtitle_map

def erase_subtitles(video_path, gpu_id=0):
    """
    Processes a video to remove subtitles using a two-pass system.
    """
    print('Starting...')

    converted_path, was_converted = convert_to_mp4(video_path)
    if not converted_path:
        return None, "Video conversion failed."

    video_path = converted_path
    video_name = os.path.basename(video_path)
    audio_path = os.path.join('Input/Audio', f"{os.path.splitext(video_name)[0]}.mp3")

    total_frames, fps, size = get_video_details(video_path)
    if total_frames == 0:
        return None, "Could not open video file."

    print(f"Video stats: {total_frames} frames, {fps:.2f} FPS, {size[0]}x{size[1]} resolution.")

    extract_audio(video_path, audio_path)

    subtitle_map = detect_all_subtitle_regions(video_path, total_frames, gpu_id)

    if not subtitle_map:
        print('No subtitles found in the input video!!!')
        return None, "No subtitles were detected in the video."

    os.makedirs('Output/Inpainted', exist_ok=True)
    inpainted_video_path = os.path.join('Output/Inpainted', video_name)
    out = cv2.VideoWriter(inpainted_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    print("\nPass 2: Inpainting subtitle regions...")
    cap = cv2.VideoCapture(video_path)

    for frame_num in tqdm(range(total_frames), desc="Inpainting frames"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in subtitle_map:
            coords = subtitle_map[frame_num]
            images = [frame]
            masks = seg_imgs(images)

            h, w = 240, 432
            new_coords, num_of_splits, final_images, final_masks = gen_regions(h, w, images, masks, coords)

            iframes, imasks = gen_frames_and_masks(final_images, final_masks)
            comp_frames = inpaint_main(iframes, imasks, gpu_id=gpu_id)
            inpainted_frame = merge(len(images), num_of_splits, new_coords, comp_frames, images)[0]
            out.write(inpainted_frame)
        else:
            # If no subtitles, just write the original frame
            out.write(frame)

    cap.release()
    out.release()
    print("Inpainting complete.")

    print('\nAdding Audio...')
    video_clip = VideoFileClip(inpainted_video_path)
    if os.path.exists(audio_path):
        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
    else:
        final_clip = video_clip

    output_video_path = os.path.join('Output', video_name)
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    if was_converted:
        print(f"Cleaning up temporary file: {converted_path}")
        os.remove(converted_path)

    print('\nCompleted :)')
    return output_video_path, "Subtitles removed successfully!"