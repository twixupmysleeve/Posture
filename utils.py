import cv2
import numpy as np


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
    print(neck)
    cv2.putText(frame, str(np.round(params[0], 2)), (int(neck[0]), int(neck[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    knee = (coords[25] + coords[26]) / 2
    print(knee)
    cv2.putText(frame, str(np.round(params[1], 2)), (int(knee[0]), int(knee[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    hip = (coords[23]+coords[24])/2
    print(hip)
    cv2.putText(frame, str(np.round(params[2], 2)), (int(hip[0]), int(hip[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    ankle = (coords[27] + coords[28]) / 2
    print(ankle)
    cv2.putText(frame, str(np.round(params[3], 2)), (int(ankle[0]), int(ankle[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    y_knee = (coords[25] + coords[26]) / 2
    print(y_knee)
    cv2.putText(frame, str(np.round(params[4], 2)), (int(y_knee[0]), int(y_knee[1]) - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def label_final_results(image, label):
    pass