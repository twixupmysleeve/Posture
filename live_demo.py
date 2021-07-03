import cv2
import mediapipe as mp
import SquatPosture as sp
import numpy as np
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# For video input:
cap = cv2.VideoCapture(0)

model = tf.keras.models.load_model("working_model_1")

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

        params = sp.get_params(results)

        if params is None:
            print("NO HUMAN!")
            continue

        flat_params = np.reshape(params, (5,1))

        # print(flat_params)

        output = model.predict(flat_params.T)

        output = output * (1/np.sum(output))

        output_name = ['c','k','h','r','x','i']

        label = ""

        for i in range(1,4):
            label += output_name[i] if output[0][i] > 0.35 else ""

        if label == "":
            label = "c"

        label += 'x' if output[0][4] > 0.1 else ''

        print(label, np.round(output[0][4],3))

        # print(output)

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
