import cv2
import os
import csv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def getVideos(path):
    videos = sorted(os.listdir(path))

    return videos

def frameNames(path):
    with open(path) as planks_csv:
        csv_data = csv.reader(planks_csv, delimiter=",")

        rows = []

        for r in csv_data:
            row = r
            rows.append((row[0], row[1], str(str(row[2])+str(row[3])+str(row[4])+str(row[2]))))

    return rows

if __name__ == "__main__":
    planks_dir = "../plank_processed"
    planks_csv_dir = "../output_vectors_plank.csv"

    planks_frames_names = frameNames(planks_csv_dir)

    counter = 0
    for vid in getVideos(planks_dir):
        cap = cv2.VideoCapture(os.path.join(planks_dir, vid))

        with mp_pose.Pose() as pose:
            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image.flags.writeable = False

                image = cv2.resize(src=image, dsize=(200, 110), interpolation=cv2.INTER_AREA)

                names = frameNames(planks_csv_dir)
                cv2.imwrite(f"{names[counter][0]}_{names[counter][1]}_{names[counter][2]}.jpeg", image)

                counter += 1

                print(f"Saved: {names[counter][0]}_{names[counter][1]}_{names[counter][2]}.png")

        cap.release()