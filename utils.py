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


def label_final_results(image, label):
    expanded_labels = {
        "c": "Correct Form",
        "k": "Knee Ahead, push your butt out",
        "h": "Back Wrongly Positioned, keep your chest up",
        "r": "Back Wrongly Positioned, keep your chest up",
        "x": "Correct Depth"
    }

    image_width, image_height, _ = image.shape

    label_list = [character for character in label]
    described_label = list(map(lambda x: expanded_labels[x], label_list))

    color = (42, 210, 48) if "c" in label_list else (13, 13, 205)

    cv2.rectangle(image,
        (0, 0), (image_height, 74),
        color,
        -1
    )

    cv2.putText(
        image, "   "+" + ".join(word for word in described_label),
        (0, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
