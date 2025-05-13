import os
import cv2


class CreateVideo:

    def __init__(self, width=640, height=360, fps=30):

        self.frames = []
        self.fps = fps
        self.width = width
        self.height = height

    def create_video(self, frames_path):

        frames_folder = sorted(os.listdir(frames_path))

        for image_name in frames_folder:

            image_path = os.path.join(frames_path, image_name)

            image = cv2.imread(image_path)
            image = cv2.resize(image, (self.width, self.height))

            if image is not None:
                self.frames.append(image)

    def save_video(self, output_path="Video.mp4"):

        if self.frames == []:
            return "No Frames to create the video"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_path, fourcc, self.fps, (self.width, self.height)
        )

        for frame in self.frames:
            video_writer.write(frame)

        video_writer.release()

        print(f"Video {output_path} saved")
