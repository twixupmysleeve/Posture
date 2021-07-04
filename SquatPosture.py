import cv2
import mediapipe as mp
import math
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

EXERCISES = [
    'squats',
    'planks',
]

def get_angle(v1, v2):
    dot = np.dot(v1, v2)
    mod_v1 = np.linalg.norm(v1)
    mod_v2 = np.linalg.norm(v2)
    cos_theta = dot/(mod_v1*mod_v2)
    theta = math.acos(cos_theta)
    return theta


def get_length(v):
    return np.dot(v, v)**0.5


def get_params(results, exercise='squats', all=False):

    if results.pose_landmarks is None:
        if exercise == 'squats':
            return np.zeros((1, 5) if not all else (19,3))
        else:
            return np.array([0, 0])

    points = {}
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    points["NOSE"] = np.array([nose.x, nose.y, nose.z])
    left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    points["LEFT_EYE"] = np.array([left_eye.x, left_eye.y, left_eye.z])
    right_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
    points["RIGHT_EYE"] = np.array([right_eye.x, right_eye.y, right_eye.z])
    mouth_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
    points["MOUTH_LEFT"] = np.array([mouth_left.x, mouth_left.y, mouth_left.z])
    mouth_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
    points["MOUTH_RIGHT"] = np.array([mouth_right.x, mouth_right.y, mouth_right.z])
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    points["LEFT_SHOULDER"] = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    points["RIGHT_SHOULDER"] = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
    left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    points["LEFT_ELBOW"] = np.array([left_elbow.x, left_elbow.y, left_elbow.z])
    right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    points["RIGHT_ELBOW"] = np.array([right_elbow.x, right_elbow.y, right_elbow.z])
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    points["RIGHT_WRIST"] = np.array([right_wrist.x, right_wrist.y, right_wrist.z])
    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    points["LEFT_WRIST"] = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    points["LEFT_HIP"] = np.array([left_hip.x, left_hip.y, left_hip.z])
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    points["RIGHT_HIP"] = np.array([right_hip.x, right_hip.y, right_hip.z])
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    points["LEFT_KNEE"] = np.array([left_knee.x, left_knee.y, left_knee.z])
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    points["RIGHT_KNEE"] = np.array([right_knee.x, right_knee.y, right_knee.z])
    left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
    points["LEFT_HEEL"] = np.array([left_heel.x, left_heel.y, left_heel.z])
    right_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
    points["RIGHT_HEEL"] = np.array([right_heel.x, right_heel.y, right_heel.z])
    left_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    points["LEFT_FOOT_INDEX"] = np.array([left_foot_index.x, left_foot_index.y, left_foot_index.z])
    right_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    points["RIGHT_FOOT_INDEX"] = np.array([right_foot_index.x, right_foot_index.y, right_foot_index.z])
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    points["LEFT_ANKLE"] = np.array([left_ankle.x, left_ankle.y, left_ankle.z])
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    points["RIGHT_ANKLE"] = np.array([right_ankle.x, right_ankle.y, right_ankle.z])

    points["MID_SHOULDER"] = (points["LEFT_SHOULDER"] + points["RIGHT_SHOULDER"]) / 2
    points["MID_HIP"] = (points["LEFT_HIP"] + points["RIGHT_HIP"]) / 2

    z_eyes = (points["RIGHT_EYE"][2] + points["LEFT_EYE"][2]) / 2
    z_mouth = (points["MOUTH_LEFT"][2] + points["MOUTH_RIGHT"][2]) / 2

    theta_neck = get_angle(np.array([0, 0, -1]),
                           points["NOSE"] - points["MID_HIP"])

    theta_s1 = get_angle(points["LEFT_ELBOW"]-points["LEFT_SHOULDER"],
                         points["LEFT_HIP"]-points["LEFT_SHOULDER"])

    theta_s2 = get_angle(points["RIGHT_ELBOW"] - points["RIGHT_SHOULDER"],
                         points["RIGHT_HIP"] - points["RIGHT_SHOULDER"])

    theta_s = (theta_s1 + theta_s2) / 2

    z_face = z_eyes - z_mouth

    theta_k1 = get_angle(points["RIGHT_HIP"] - points["RIGHT_KNEE"],
                         points["RIGHT_ANKLE"] - points["RIGHT_KNEE"])

    theta_k2 = get_angle(points["LEFT_HIP"] - points["LEFT_KNEE"],
                         points["LEFT_ANKLE"] - points["LEFT_KNEE"])

    theta_k = (theta_k1 + theta_k2) / 2

    theta_h1 = get_angle(points["RIGHT_KNEE"] - points["RIGHT_HIP"],
                         points["RIGHT_SHOULDER"] - points["RIGHT_HIP"])

    theta_h2 = get_angle(points["LEFT_KNEE"] - points["LEFT_HIP"],
                         points["LEFT_SHOULDER"] - points["LEFT_HIP"])

    theta_h = (theta_h1 + theta_h2) / 2

    torso_length = get_length(points['MID_SHOULDER'] - points['MID_HIP'])
    left_thigh_length = get_length(points['LEFT_KNEE'] - points['LEFT_HIP'])
    right_thigh_length = get_length(points['RIGHT_KNEE'] - points['RIGHT_HIP'])
    left_tibula_length = get_length(points['LEFT_KNEE'] - points['LEFT_HEEL'])
    right_tibula_length = get_length(points['RIGHT_KNEE'] - points['RIGHT_HEEL'])

    thigh_length = (left_thigh_length + right_thigh_length) / 2
    tibula_length = (left_tibula_length + right_tibula_length) / 2

    length_normalization_factor = (1 / (tibula_length))**0.5

    z1 = (points["RIGHT_ANKLE"][2] + points["RIGHT_HEEL"][2]) / 2 - points["RIGHT_FOOT_INDEX"][2]

    z2 = (points["LEFT_ANKLE"][2] + points["LEFT_HEEL"][2]) / 2 - points["LEFT_FOOT_INDEX"][2]

    z = (z1 + z2) / 2

    z *= length_normalization_factor

    left_foot_y = (points["LEFT_ANKLE"][1] + points["LEFT_HEEL"][1] + points["LEFT_FOOT_INDEX"][1]) / 3
    right_foot_y = (points["RIGHT_ANKLE"][1] + points["RIGHT_HEEL"][1] + points["RIGHT_FOOT_INDEX"][1]) / 3

    left_ky = points["LEFT_KNEE"][1] - left_foot_y
    right_ky = points["RIGHT_KNEE"][1] - right_foot_y

    ky = (left_ky + right_ky) / 2

    ky *= length_normalization_factor

    left_foot = points["LEFT_HEEL"] - points["LEFT_FOOT_INDEX"]
    theta_left_foot = get_angle(left_foot, np.array([left_foot[0], left_foot[1], points["LEFT_FOOT_INDEX"][2]]))
    right_foot = points["RIGHT_HEEL"] - points["RIGHT_FOOT_INDEX"]
    theta_right_foot = get_angle(right_foot, np.array([right_foot[0], right_foot[1], points["RIGHT_FOOT_INDEX"][2]]))

    theta_foot = (theta_right_foot + theta_left_foot) / 2

    if exercise=='squats':
        params = np.array([theta_neck, theta_k, theta_h, z, ky])
    elif exercise=='plank':
        params = np.array([theta_s1, theta_s2])

    if all:
        params = np.array([[x, y, z] for pos, (x, y, z) in points.items()]) * length_normalization_factor

    return np.round(params, 2)
