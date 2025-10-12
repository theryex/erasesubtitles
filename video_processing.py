import os
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip

from preprocessing import gen_image_frames, seg_imgs
from detectText import get_coords
from splitRegion import gen_regions
from inpaint import gen_frames_and_masks, inpaint_main, merge

def erase_subtitles(video_name, save_fps=30):
    """
    Processes a video to remove subtitles.

    Args:
        video_name (str): The name of the video file.
        save_fps (int): The frame rate for the output video.

    Returns:
        tuple: A tuple containing the path to the output video and a status message.
    """
    print('Starting...')
    print('\nPreprocessing...')

    video_path = os.path.join('Input/Video', video_name)
    audio_path = os.path.join('Input/Audio', f"{os.path.splitext(video_name)[0]}.mp3")

    images = gen_image_frames(video_path, audio_path)
    if not images:
        return None, "Could not generate image frames from the video."

    masks = seg_imgs(images)
    num_of_frames = len(masks)

    print('Stats of the video')
    print('Number of frames:', num_of_frames)

    print('\nDetecting Text...')
    coords = get_coords(num_of_frames, masks)
    print('Subtitle Region coords:', coords)
    print('Subtitle Text Detection Done')

    if coords:
        h, w = 240, 432
        print(f'\nSplitting subtitle region into {h}x{w} parts...')
        new_coords, num_of_splits, final_images, final_masks = gen_regions(h, w, images, masks, coords)
        print('Split coords:', new_coords)
        print('Splitting Concluded')

        print('\nInpainting Begins...')
        iframes, imasks = gen_frames_and_masks(final_images, final_masks)
        comp_frames = inpaint_main(iframes, imasks)
        inpainted_frames = merge(num_of_frames, num_of_splits, new_coords, comp_frames, images)
        print('Inpainting Complete')

        im_h, im_w = inpainted_frames[0].shape[:2]
        size = (im_w, im_h)

        # Ensure output directories exist
        os.makedirs('Output/Inpainted', exist_ok=True)
        os.makedirs('Output', exist_ok=True)

        inpainted_video_path = os.path.join('Output/Inpainted', video_name)
        out = cv2.VideoWriter(inpainted_video_path, cv2.VideoWriter_fourcc(*"mp4v"), save_fps, size)

        print(f'\nStoring video in {inpainted_video_path}')
        for frame in inpainted_frames:
            out.write(frame)
        out.release()

        print('\nAdding Audio')
        video_clip = VideoFileClip(inpainted_video_path)

        if os.path.exists(audio_path):
            audio_clip = AudioFileClip(audio_path)
            final_clip = video_clip.set_audio(audio_clip)
        else:
            final_clip = video_clip

        output_video_path = os.path.join('Output', video_name)
        final_clip.write_videofile(output_video_path)

        return output_video_path, "Subtitles removed successfully!"
    else:
        print('No Subtitles found in the input video!!!')
        return None, "No subtitles were detected in the video."

    print('\nCompleted :)')