# Importing Libraries
import os
import cv2
import numpy as np
import moviepy.editor as mp


# To generate image frames and audio from the video
def gen_image_frames(video_path, audio_path):
    """
    Generates image frames and extracts audio from a video file.

    Args:
        video_path (str): The path to the video file.
        audio_path (str): The path to save the extracted audio.

    Returns:
        list: A list of image frames from the video.
    """
    audio_dir = os.path.dirname(audio_path)
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    vidcap = cv2.VideoCapture(video_path)

    imgs = []
    success, image = vidcap.read()
    while success:
        imgs.append(image)
        success, image = vidcap.read()

    try:
        my_clip = mp.VideoFileClip(video_path)
        my_clip.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Could not extract audio: {e}")

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