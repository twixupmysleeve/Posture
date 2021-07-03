import cv2
import mediapipe as mp
import SquatPosture as sp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def landmarks_list_to_array(landmark_list, image_shape):
    rows, cols, _ = image_shape

    if landmark_list is None:
        return None

    return np.asarray([(lmk.x * cols, lmk.y * rows)
                       for lmk in landmark_list.landmark])


def label_params(frame, params, coords):

    if coords is None:
        return

    params = params * 180/3.14159265

    neck = (coords[11]+coords[12])/2
    # print(neck)
    cv2.putText(frame, str(np.round(params[0], 2)), (int(neck[0]), int(neck[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    knee = (coords[25] + coords[26]) / 2
    # print(knee)
    cv2.putText(frame, str(np.round(params[1], 2)), (int(knee[0]), int(knee[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    hip = (coords[23]+coords[24])/2
    # print(hip)
    cv2.putText(frame, str(np.round(params[2], 2)), (int(hip[0]), int(hip[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    ankle = (coords[27] + coords[28]) / 2
    # print(ankle)
    cv2.putText(frame, str(np.round(params[3], 2)), (int(ankle[0]), int(ankle[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    y_knee = (coords[25] + coords[26]) / 2
    # print(y_knee)
    cv2.putText(frame, str(np.round(params[4], 2)), (int(y_knee[0]), int(y_knee[1]) - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


directory = 'data/processed/'

# video_names = sorted(os.listdir(directory))

videos_to_use = ["000","001","002","003","004","005","006","007","008","009","010","023"]
video_names = [video + "_squat.mp4" for video in videos_to_use]

file = open("data/input_vectors.csv", "w")

for video_name in video_names:
    cap = cv2.VideoCapture("data/processed/" + video_name)
    frame_number = 0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                # continue
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image_hight, image_width, _ = image.shape

            params = sp.get_params(results)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            coords = landmarks_list_to_array(results.pose_landmarks, image.shape)
            label_params(image, params, coords)

            file.write("{},{},{},{},{},{},{}\n".format(
                video_name[0:3],
                frame_number,
                params[0],
                params[1],
                params[2],
                params[3],
                params[4]
            ))
            file.flush()
            frame_number += 1

            cv2.imshow('MediaPipe Pose', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    print(video_name)

file.close()

cv2.destroyAllWindows()
