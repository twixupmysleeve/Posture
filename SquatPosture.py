import cv2
import mediapipe as mp
import math


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def get_angle(v1, v2):
    dot = (v1[0] * v2[0]) + (v1[1] * v2[1]) + (v1[2] * v2[2])
    mod_v1 = ((v1[0] ** 2) + (v1[1] ** 2) + (v1[2] ** 2)) ** 0.5
    mod_v2 = ((v2[0] ** 2) + (v2[1] ** 2) + (v2[2] ** 2)) ** 0.5
    cos_theta = dot/(mod_v1*mod_v2)
    theta = math.acos(cos_theta)
    return theta


class SquatBody:

    def __init__(self, results):
        self.points = {}
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        self.points["NOSE"] = nose.x, nose.y, nose.z
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        self.points["LEFT_SHOULDER"] = left_shoulder.x, left_shoulder.y, left_shoulder.z
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        self.points["RIGHT_SHOULDER"] = right_shoulder.x, right_shoulder.y, right_shoulder.z
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        self.points["LEFT_HIP"] = left_hip.x, left_hip.y, left_hip.z
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        self.points["RIGHT_HIP"] = right_hip.x, right_hip.y, right_hip.z
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        self.points["LEFT_KNEE"] = left_knee.x, left_knee.y, left_knee.z
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        self.points["RIGHT_KNEE"] = right_knee.x, right_knee.y, right_knee.z
        left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        self.points["LEFT_HEEL"] = left_heel.x, left_heel.y, left_heel.z
        right_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
        self.points["RIGHT_HEEL"] = right_heel.x, right_heel.y, right_heel.z
        left_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        self.points["LEFT_FOOT_INDEX"] = left_foot_index.x, left_foot_index.y, left_foot_index.z
        right_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        self.points["RIGHT_FOOT_INDEX"] = right_foot_index.x, right_foot_index.y, right_foot_index.z
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        self.points["LEFT_ANKLE"] = left_ankle.x, left_ankle.y, left_ankle.z
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        self.points["RIGHT_ANKLE"] = right_ankle.x, right_ankle.y, right_ankle.z

        self.theta_neck = angle
