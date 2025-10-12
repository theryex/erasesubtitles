import streamlit as st
import os
from video_processing import erase_subtitles

def main():
    st.title("Multilingual Subtitle Text Removal")

    # Create necessary directories if they don't exist
    os.makedirs("Input/Video", exist_ok=True)
    os.makedirs("Input/Audio", exist_ok=True)
    os.makedirs("Output", exist_ok=True)

    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"])

    if video_file:
        st.video(video_file)

        if st.button("Remove Subtitles"):
            with st.spinner("Processing..."):
                # Save the uploaded file to a temporary location
                input_video_path = os.path.join("Input/Video", video_file.name)
                with open(input_video_path, "wb") as f:
                    f.write(video_file.getbuffer())

                # Process the video to remove subtitles
                output_video_path, message = erase_subtitles(video_file.name)

                if output_video_path:
                    st.success(message)
                    # Display the processed video
                    with open(output_video_path, "rb") as f:
                        st.video(f)
                else:
                    st.error(message)

if __name__ == "__main__":
    main()