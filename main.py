import json

import cv2
from PIL import Image
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from mtcnn import MTCNN

from services.face_detection import FaceDetection
from services.vgg16_model import load_metadata
from utils import find_best_solution


def vgg16_model():
    img = image.load_img(
        "resources/gray/val/adriana_lima/adriana_lima1.jpg",
        target_size=(224, 224),
    )
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    saved_model = load_model("resources/gray/vgg16_gray.h5")
    output = saved_model.predict(img)
    _index, _max = find_best_solution(output)
    print("_max", output, _max, _index)
    num_classes, class_indices, labels = load_metadata("resources/metadata.json")
    print(labels[str(_index)])


def detection():
    face_detector = MTCNN()
    rgb_img, faces = FaceDetection.detect("resources/g7.jpg", face_detector)
    for i, face in enumerate(faces):
        x, y, w, h = face["box"]
        cv2.rectangle(
            rgb_img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2
        )
        left_eye = face["keypoints"]["left_eye"]
        right_eye = face["keypoints"]["right_eye"]
        nose = face["keypoints"]["nose"]
        mouth_left = face["keypoints"]["mouth_left"]
        mouth_right = face["keypoints"]["mouth_right"]
        rgb_img = cv2.circle(
            rgb_img, left_eye, radius=0, color=(0, 0, 255), thickness=8
        )
        rgb_img = cv2.circle(
            rgb_img, right_eye, radius=0, color=(0, 0, 255), thickness=8
        )
        rgb_img = cv2.circle(rgb_img, nose, radius=0, color=(0, 0, 255), thickness=8)
        rgb_img = cv2.circle(
            rgb_img, mouth_left, radius=0, color=(0, 0, 255), thickness=8
        )
        rgb_img = cv2.circle(
            rgb_img, mouth_right, radius=0, color=(0, 0, 255), thickness=8
        )
    plt.imshow(Image.fromarray(rgb_img))
    plt.show()


if __name__ == "__main__":
    file = open("resources/test.json")
    data = json.load(file)
    file.close()
    precision_result = data["precision_result"]
    recall_result = data["recall_result"]
    counter = 0
    total = 0
    for key, value in precision_result.items():
        counter += 1
        total += value
    print(counter, total, total / counter)
    counter = 0
    total = 0
    for key, value in recall_result.items():
        counter += 1
        total += value
    print(counter, total, total / counter)
