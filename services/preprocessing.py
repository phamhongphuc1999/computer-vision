import cv2
import os
import numpy as np
from mtcnn import MTCNN
from typing import List
from services.face_detection import FaceDetection


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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, faces = FaceDetection.detect(image_path, face_detector)
            if len(faces) > 0:
                x, y, w, h = faces[0]["box"]
                face_roi = img[y : y + h, x : x + w]
                resized_face = cv2.resize(face_roi, (224, 224))
                folder_name = image_path.split("/")[-2]
                names = folder_name[5:].lower().split(" ")
                name = "_".join(names)
                output_folder = os.path.join(output_directory, name)
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f"{name}{i}.jpg")
                cv2.imwrite(output_path, resized_face)
                processed_images.append(resized_face)
            else:
                images_without_faces.append(image_path)
        print(
            f"\nImages without faces detected: {images_without_faces}({(len(images_without_faces))}/{len(image_paths)})"
        )
        return np.array(processed_images)
