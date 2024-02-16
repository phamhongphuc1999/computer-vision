import json

import cv2
import os

import face_recognition
import numpy as np
from mtcnn import MTCNN
from typing import List

from services.face_detection import FaceDetection
from utils import NumpyArrayEncoder


class Preprocessing:
    @staticmethod
    def get_image_paths(data_directory: str):
        all_image_paths = []
        for folder_name in os.listdir(data_directory):
            folder_path = os.path.join(data_directory, folder_name)
            if os.path.isdir(folder_path):
                image_paths = [
                    os.path.join(folder_path, image_name)
                    for image_name in os.listdir(folder_path)
                ]
                all_image_paths.extend(image_paths)
        return all_image_paths

    @staticmethod
    def detect_and_save(image_paths: List[str], output_directory: str):
        processed_images = []
        images_without_faces = []
        face_detector = MTCNN()
        for i, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            _, faces = FaceDetection.detect(image_path, face_detector)
            if len(faces) > 0:
                x, y, w, h = faces[0]["box"]
                face_roi = img[y : y + h, x : x + w]
                resized_face = cv2.resize(face_roi, (224, 224))
                folder_name = image_path.split("/")[-2]
                output_folder = os.path.join(output_directory, folder_name)
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f"detected_face_{i}.jpg")
                cv2.imwrite(output_path, resized_face)
                processed_images.append(resized_face)
            else:
                images_without_faces.append(image_path)

            if i % 50 == 0:
                print(f"{i}/{len(image_paths)} images processed", end="\r", flush=True)
        print(
            f"\nImages without faces detected: {images_without_faces}({(len(images_without_faces))}/{len(image_paths)})"
        )
        return np.array(processed_images)

    @staticmethod
    def face_encodings(image_paths: List[str], output_directory: str):
        result = {}
        for i, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(
                rgb_image, known_face_locations=[(0, 224, 224, 0)]
            )
            folder_name = image_path.split("/")[-2]
            name = folder_name
            if name not in result:
                result[name] = []
            result[name].append(encoding[0])
            print(f"{i}, {image_path}")
            if i > 500:
                break
        for name in result:
            src_path = os.path.join(output_directory, f"{name}.json")
            file = open(src_path, "w", encoding="utf-8")
            json.dump(result[name], file, cls=NumpyArrayEncoder)
            file.close()
