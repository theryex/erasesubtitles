import os
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
import math

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

def erase_subtitles(video_name):
    """
    Processes a video to remove subtitles in chunks to handle large files.
    """
    print('Starting...')
    video_path = os.path.join('Input/Video', video_name)
    audio_path = os.path.join('Input/Audio', f"{os.path.splitext(video_name)[0]}.mp3")

    # Get video properties
    total_frames, fps, size = get_video_details(video_path)
    if total_frames == 0:
        return None, "Could not open video file."

    print(f"Video stats: {total_frames} frames, {fps:.2f} FPS, {size[0]}x{size[1]} resolution.")

    # Extract audio once
    print("\nExtracting audio...")
    extract_audio(video_path, audio_path)

    # Prepare for subtitle detection
    print("\nDetecting subtitle regions...")
    cap = cv2.VideoCapture(video_path)
    sample_frames = []
    for i in range(min(total_frames, 300)):  # Sample first 300 frames
        ret, frame = cap.read()
        if not ret:
            break
        sample_frames.append(frame)
    cap.release()

    masks = seg_imgs(sample_frames)
    coords = get_coords(len(masks), masks)
    print('Subtitle Region coords:', coords)

    if not coords:
        print('No subtitles found in the input video!!!')
        return None, "No subtitles were detected in the video."

    # Prepare video writer
    os.makedirs('Output/Inpainted', exist_ok=True)
    inpainted_video_path = os.path.join('Output/Inpainted', video_name)
    out = cv2.VideoWriter(inpainted_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    print("\nStarting chunk-based inpainting...")
    for start_frame in range(0, total_frames, CHUNK_SIZE):
        end_frame = min(start_frame + CHUNK_SIZE, total_frames)
        print(f"Processing frames from {start_frame} to {end_frame-1}...")

        # Generate frames for the current chunk
        images = gen_image_frames(video_path, start_frame, end_frame)
        if not images:
            continue

        masks = seg_imgs(images)

        h, w = 240, 432
        new_coords, num_of_splits, final_images, final_masks = gen_regions(h, w, images, masks, coords)

        iframes, imasks = gen_frames_and_masks(final_images, final_masks)
        comp_frames = inpaint_main(iframes, imasks)
        inpainted_frames = merge(len(images), num_of_splits, new_coords, comp_frames, images)

        for frame in inpainted_frames:
            out.write(frame)

    out.release()
    print("Chunk-based inpainting complete.")

    print('\nAdding Audio...')
    video_clip = VideoFileClip(inpainted_video_path)
    if os.path.exists(audio_path):
        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
    else:
        final_clip = video_clip  # No audio to attach

    output_video_path = os.path.join('Output', video_name)
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    print('\nCompleted :)')
    return output_video_path, "Subtitles removed successfully!"