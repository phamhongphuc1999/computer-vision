import math
import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
from utils import find_best_solution
from services.vgg16_model import load_metadata
from services.face_detection import FaceDetection


class AppService:
    def __init__(self):
        self._face_detector = MTCNN()
        self._face_recognition = load_model("resources/vgg16_1.h5")
        num_classes, class_indices, labels = load_metadata("resources/metadata.json")
        self.num_classes = num_classes
        self.class_indices = class_indices
        self.labels = labels

    def _recognition(self, face_roi):
        resized = cv2.resize(face_roi, (224, 224))
        np_arr = np.array(resized)
        img_arr = np.asarray(np_arr)
        final_img = np.expand_dims(img_arr, axis=0)
        output = self._face_recognition.predict(final_img)
        index, percent = find_best_solution(output)
        predicted_name = self.labels[str(index)]
        return predicted_name, percent

    def analytic(self, path: str, display_width: int, show_mark=False):
        _, faces = FaceDetection.detect(path, self._face_detector)
        image = cv2.imread(path)
        face_locations = []
        height, width, channels = image.shape
        new_height = math.floor(display_width * height / width)

        width_ratio = display_width / width
        height_ratio = new_height / height
        for i, face in enumerate(faces):
            x, y, w, h = face["box"]
            face_roi = image[y : y + h, x : x + w]
            predicted_name, percent = self._recognition(face_roi)
            face_locations.append(
                {
                    "x": x * width_ratio,
                    "y": y * height_ratio,
                    "w": w * width_ratio,
                    "h": h * height_ratio,
                    "predicted_name": predicted_name,
                    "percent": percent,
                    "id": i,
                }
            )
            cv2.rectangle(
                image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 1
            )
            if show_mark:
                left_eye = face["keypoints"]["left_eye"]
                right_eye = face["keypoints"]["right_eye"]
                nose = face["keypoints"]["nose"]
                mouth_left = face["keypoints"]["mouth_left"]
                mouth_right = face["keypoints"]["mouth_right"]
                image = cv2.circle(
                    image, left_eye, radius=0, color=(0, 0, 255), thickness=3
                )
                image = cv2.circle(
                    image, right_eye, radius=0, color=(0, 0, 255), thickness=3
                )
                image = cv2.circle(
                    image, nose, radius=0, color=(0, 0, 255), thickness=3
                )
                image = cv2.circle(
                    image, mouth_left, radius=0, color=(0, 0, 255), thickness=3
                )
                image = cv2.circle(
                    image, mouth_right, radius=0, color=(0, 0, 255), thickness=3
                )
        image = cv2.resize(image, (display_width, new_height))
        return image, face_locations, new_height
