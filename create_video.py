"""
Create Video from Frames

This module provides functionality to create a video file from a sequence of image frames.
It's useful for converting output from image processing algorithms into a playable video
or for creating timelapses from a series of captured images.
"""

import os
import cv2


class CreateVideo:
    """
    A class to create videos from a sequence of image frames.

    This class provides methods to load image frames from a directory,
    optionally resize them to a standard size, and save them as a video file
    using the specified codec and frame rate.
    """

    def __init__(self, width=640, height=360, fps=30):
        """
        Initialize the video creator with specified parameters.

        Args:
            width (int): Output video width in pixels (default: 640)
            height (int): Output video height in pixels (default: 360)
            fps (int): Frames per second for the output video (default: 30)
        """
        self.frames = []  # List to store loaded video frames
        self.fps = fps  # Frame rate of the output video
        self.width = width  # Width dimension of the output video
        self.height = height  # Height dimension of the output video

    def create_video(self, frames_path):
        """
        Load image frames from a directory and prepare them for video creation.

        The method:
        1. Lists and sorts files in the specified directory
        2. Loads each image and resizes it to the specified dimensions
        3. Stores valid frames in the frames list

        Args:
            frames_path (str): Path to directory containing image frames
                             Files should be named in a way that sorts correctly
                             (e.g., frame_001.png, frame_002.png, etc.)
        """
        # Get list of image files sorted by name
        frames_folder = sorted(os.listdir(frames_path))

        for image_name in frames_folder:
            # Create the full path to the image
            image_path = os.path.join(frames_path, image_name)

            # Load the image and resize to the specified dimensions
            image = cv2.imread(image_path)
            image = cv2.resize(image, (self.width, self.height))

            # Add valid images to the frames list
            if image is not None:
                self.frames.append(image)

    def save_video(self, output_path="Video.mp4"):
        """
        Create a video file from the loaded frames.

        Args:
            output_path (str): Path for the output video file (default: "Video.mp4")

        Returns:
            str: Error message if no frames are available, otherwise None
        """
        if self.frames == []:
            return "No Frames to create the video"

        # Initialize the video writer with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
        video_writer = cv2.VideoWriter(
            output_path, fourcc, self.fps, (self.width, self.height)
        )

        # Write each frame to the video
        for frame in self.frames:
            video_writer.write(frame)

        # Release the video writer to finalize the file
        video_writer.release()

        print(f"Video {output_path} saved")
