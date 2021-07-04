import os
# import cv2
# import mediapipe
import ffmpeg
import pickle

def preprocess(video, output, fps=12, width=720, height=1280, exercise='squats'):
    stream = ffmpeg.input(video)
    stream = stream.filter('fps', fps=fps, round='up')
    if exercise=='squats':
        out = "processed/"+output
    elif exercise=='plank':
        out = "plank_processed/"+output

    stream = ffmpeg.output(stream, out)

    ffmpeg.run(stream, quiet=True)


if __name__ == "__main__":
    processed = sorted(os.listdir("plank_processed"))
    raw = sorted(os.listdir("plank_raw"))

    if len(processed) == 0:
        count = 0
    else:
        count = int(processed[-1][:3])
        count += 1

    print(count)

    for i in raw:
        file = f"plank_raw/{i}"
        leading_count = str(count).zfill(3)
        name = leading_count+"_plank.mp4"
        preprocess(file, name)

        count+=1
        print(name)

        os.remove(file)
        # break