import cv2
import mediapipe as mp
import SquatPosture as sp
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import *
from csv import writer

csv_file = open('plotting_live.csv', 'w+')
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
writer_object = writer(csv_file)
writer_object.writerow(['neck', 'knee', 'hip', 'ankle', 'y-knee'])

# For video input:
cap = cv2.VideoCapture(0)

model = tf.keras.models.load_model("working_model_1")
counter_for_renewal = 0
with mp_pose.Pose() as pose:
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

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        params = sp.get_params(results, all=True)

        if params is None:
            print("NO HUMAN!")
            continue

        flat_params = np.reshape(params, (57, 1))

        #if counter_for_renewal > 100:
            #csv_file.truncate(1)
        # writer_object.writerow(flat_params.T.flatten())
        # csv_file.flush()

        # counter_for_renewal += 1
        # print(flat_params)

        output = model.predict(flat_params.T)

        # output[0][2] *= 5
        # output[0][4] *= 3

        output = output * (1 / np.sum(output))

        output_name = ['c', 'k', 'h', 'r', 'x', 'i']

        label = ""

        print(output)

        for i in range(1, 5):
            label += output_name[i] if output[0][i] > 0.1 else ""

        if label == "":
            label = "c"

        # print(label, output)

        label_final_results(image, label)

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
csv_file.close()
cap.release()
cv2.destroyAllWindows()
