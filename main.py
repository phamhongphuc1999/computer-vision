import cv2
from PIL import Image
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from mtcnn import MTCNN

from services.evaluation import Evaluation
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
    img = cv2.imread("dataset/pins_barack obama/barack obama120_636.jpg")
    _, faces = FaceDetection.detect(
        "dataset/pins_barack obama/barack obama120_636.jpg", face_detector
    )
    for i, face in enumerate(faces):
        x, y, w, h = face["box"]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        left_eye = face["keypoints"]["left_eye"]
        right_eye = face["keypoints"]["right_eye"]
        nose = face["keypoints"]["nose"]
        mouth_left = face["keypoints"]["mouth_left"]
        mouth_right = face["keypoints"]["mouth_right"]
        img = cv2.circle(img, left_eye, radius=0, color=(0, 0, 255), thickness=8)
        img = cv2.circle(img, right_eye, radius=0, color=(0, 0, 255), thickness=8)
        img = cv2.circle(img, nose, radius=0, color=(0, 0, 255), thickness=8)
        img = cv2.circle(img, mouth_left, radius=0, color=(0, 0, 255), thickness=8)
        img = cv2.circle(img, mouth_right, radius=0, color=(0, 0, 255), thickness=8)
    plt.imshow(Image.fromarray(img))
    plt.show()


if __name__ == "__main__":
    precision, recall, f1 = Evaluation.metrics()
    x = ["Precision", "Recall"]
    y = [precision, recall]
    plt.bar(x, y, width=0.1)
    for index, value in enumerate(y):
        plt.text(index - 0.035, value + 0.01, str(float("{:.2f}".format(value))))
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1))
    plt.show()
