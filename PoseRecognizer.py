# 使用Mediapipe
import math

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


class PoseRecognizer:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    # landmarks_dict = {e.name: e.value for e in mp.solutions.hands.HandLandmark}  # 获得关键点名称和索引的字典

    # 图像上部的点y  < 下部的点的y

    Visibility_Thresholding=0.8
    @staticmethod
    def pointsClosed(x1, y1, x2, y2, threshold=0.07):
        '''返回两个2D点是否足够近'''
        distance = math.dist([x1, y1], [x2, y2])
        # print(distance)
        if distance < threshold:
            return True
        else:
            return False

    @staticmethod
    def isCrossedWrists(list_landmarks: landmark_pb2.NormalizedLandmarkList):
        kps = list_landmarks.landmark
        #两个手掌的标记挨在一起，15，16
        if PoseRecognizer.pointsClosed(kps[15].x, kps[15].y, kps[16].x, kps[16].y):
            return True
        else:
            return False

    @staticmethod
    def hasEnoughMarks(list_landmarks: landmark_pb2.NormalizedLandmarkList):
        kps = list_landmarks.landmark
        if kps[32].visibility>PoseRecognizer.Visibility_Thresholding:
            print("Ok!")
        pass


if __name__ == '__main__':
    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles
    # mp_pose = mp.solutions.pose

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with PoseRecognizer.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            pose_landmarks: landmark_pb2.NormalizedLandmarkList=results.pose_landmarks
            if pose_landmarks :#判断一下是否得到了pose
                # print(pose_landmarks.landmark[0])#
                # print(PoseRecognizer.isCrossedWrists(pose_landmarks))#测试一下手腕交叉
                PoseRecognizer.hasEnoughMarks(pose_landmarks)
            else:
                # print("No People!")
                pass
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            PoseRecognizer.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                PoseRecognizer.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=PoseRecognizer.mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    pass
