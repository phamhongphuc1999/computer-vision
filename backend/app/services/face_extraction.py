from typing import List

from imutils import paths
import face_recognition
import pickle
import cv2
import os

from app.services import convert_xywh_to_recognition
from app.services.face_detection import FaceDetection


class FaceExtraction:
    @staticmethod
    def extract_by_file(img_path: str):
        known_encodings = []
        known_names = []
        name = img_path.split(os.path.sep)[-2]
        image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = FaceDetection.face_locations(image_path=img_path)
        face_locations = map(convert_xywh_to_recognition, face_locations)
        encodings = face_recognition.face_encodings(
            face_image=rgb_image, known_face_locations=face_locations
        )
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)
        return known_encodings, known_names

    @staticmethod
    def extract_by_folder(folder_path: str):
        img_paths = list(paths.list_images(folder_path))
        known_encodings = []
        known_names = []
        for (i, img_path) in enumerate(img_paths):
            encodings, names = FaceExtraction.extract_by_file(img_path)
            known_encodings = known_encodings + encodings
            known_names = known_names + names
        return known_encodings, known_names

    @staticmethod
    def save(file_path: str, encodings: List, names: List):
        data = {"encodings": encodings, "names": names}
        f = open(file_path, "wb")
        f.write(pickle.dumps(data))
        f.close()
