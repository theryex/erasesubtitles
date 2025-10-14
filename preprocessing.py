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

def gen_image_frames(video_path, start_frame, end_frame):
    """
    Generates image frames from a specific segment of a video file.

    Args:
        video_path (str): The path to the video file.
        start_frame (int): The starting frame number.
        end_frame (int): The ending frame number.

    Returns:
        list: A list of image frames from the specified video segment.
    """
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print("Error: Could not open video.")
        return []

    imgs = []
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while current_frame < end_frame:
        success, image = vidcap.read()
        if not success:
            break
        imgs.append(image)
        current_frame += 1

    vidcap.release()
    return imgs


# Color segmentation
def seg(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([150, 15, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

def seg_imgs(images):
    masks = [None for i in range(len(images))]
    for i in range(len(images)):
        masks[i] = seg(images[i])
    return masks