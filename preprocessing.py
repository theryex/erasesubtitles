# Importing Libraries
import os
import cv2
import numpy as np
import moviepy.editor as mp


def extract_audio(video_path, audio_path):
    """Extracts audio from a video file."""
    if os.path.exists(audio_path):
        print("Audio file already exists. Skipping extraction.")
        return

    try:
        print(f"Extracting audio to {audio_path}")
        video_clip = mp.VideoFileClip(video_path)
        audio_clip = video_clip.audio
        if audio_clip:
            audio_clip.write_audiofile(audio_path, codec='mp3')
            audio_clip.close()
        video_clip.close()
    except Exception as e:
        print(f"Could not extract audio: {e}")

def gen_image_frames(video_path, start_frame=None, end_frame=None):
    """
    Generates image frames from a video file.
    Can be used to extract a specific segment if start and end frames are provided.
    """
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print("Error: Could not open video.")
        return []

    imgs = []
    if start_frame is not None:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
    else:
        current_frame = 0

    while True:
        success, image = vidcap.read()
        if not success:
            break

        imgs.append(image)
        current_frame += 1

        if end_frame is not None and current_frame >= end_frame:
            break

    vidcap.release()
    return imgs


def seg(image):
    """
    Creates a binary mask of potential text regions using color segmentation
    followed by morphological closing to handle outlined text.
    """
    # Isolate white-like pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([180, 55, 255]) # Loosened the constraints slightly
    mask = cv2.inRange(hsv, lower, upper)

    # Use morphological closing to fill gaps in the text
    # This helps connect letters that have black outlines
    kernel = np.ones((10,10),np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Convert single-channel mask back to 3-channel BGR for consistency
    final_mask = cv2.cvtColor(closed_mask, cv2.COLOR_GRAY2BGR)

    return final_mask

def seg_imgs(images):
    masks = [None for i in range(len(images))]
    for i in range(len(images)):
        masks[i] = seg(images[i])
    return masks