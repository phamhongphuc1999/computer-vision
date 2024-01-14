import os
from typing import Sequence, List

import cv2
import dlib
from imutils import face_utils

from app.services import rotate_image, draw_xywh_image, get_img

face_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(face_path)

eye_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
eye_cascade = cv2.CascadeClassifier(eye_path)


class FaceDetection:
    ANGLE_STEP = 10

    @staticmethod
    def face_locations(image_path: str = None, image=None):
        real_image = get_img(image_path=image_path, image=image)
        gray_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces

    @staticmethod
    def rotated_face_locations(image_path: str = None, image=None):
        real_image = get_img(image_path=image_path, image=image)
        angle = 0
        gray_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
        while angle <= 360:
            rotated_image = rotate_image(gray_image, angle)
            faces = face_cascade.detectMultiScale(
                rotated_image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(10, 10),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces) > 0:
                return faces, angle
            angle += FaceDetection.ANGLE_STEP
        return [], 0

    @staticmethod
    def landmark(image_path: str = None, image=None):
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces_location = FaceDetection.face_locations(image_path=image_path, image=image)
        real_image = get_img(image_path=image_path, image=image)
        gray_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
        result = []
        for x, y, w, h in faces_location:
            d_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = predictor(gray_image, d_rect)
            points = face_utils.shape_to_np(landmarks)
            result.append(((x, y, w, h), points))
        return result

    @staticmethod
    def rotated_landmark(image_path: str = None, image=None):
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces_location, angle = FaceDetection.rotated_face_locations(image_path=image_path, image=image)
        real_image = get_img(image_path=image_path, image=image)
        gray_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
        result = []
        for x, y, w, h in faces_location:
            d_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = predictor(gray_image, d_rect)
            points = face_utils.shape_to_np(landmarks)
            result.append(((x, y, w, h), points, angle))
        return result

    @staticmethod
    def eyes(image_path: str = None, image=None):
        faces_location = FaceDetection.face_locations(image_path=image_path, image=image)
        real_image = get_img(image_path=image_path, image=image)
        gray_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
        result = []
        for (x, y, w, h) in faces_location:
            roi_gray = gray_image[y: y + h, x: x + w]
            roi_color = gray_image[y: y + h, x: x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            result.append(((x, y, w, h), roi_color, eyes))
        return result

    @staticmethod
    def draw(image, face_location: Sequence[int]):
        for location in face_location:
            draw_xywh_image(image, location)

    @staticmethod
    def draw_landmark(image, data: List, mode: 'normal' or 'rotate'):
        for item in data:
            if mode == 'normal':
                face_location, points = item
            else:
                face_location, points, angle = item
            for i in points:
                draw_xywh_image(image, face_location)
                cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), -1)

    @staticmethod
    def draw_eyes(image, data: List):
        for item in data:
            face_location, roi_color, eyes = item
            draw_xywh_image(image, face_location)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
