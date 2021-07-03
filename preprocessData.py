import cv2
import mediapipe
import ffmpeg

def preprocess(video, fps=12):
    stream = ffmpeg.input(video)
    stream=stream.filter('fps', fps=fps, round='up')
    stream = ffmpeg.output(stream, video)

    ffmpeg.run(stream, quiet=True)

