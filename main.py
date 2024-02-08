import cv2
import os

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from mtcnn import MTCNN

from services.face_detection import FaceDetection
from services.face_embedding import FaceEmbedding
from services.vgg16_model import get_dataset
from utils import count_file_and_folder
from keras.preprocessing import image
import numpy as np
from keras.models import load_model


def my_fun():
    device = "cpu"
    face_detector = MTCNN()
    face_cascade = cv2.CascadeClassifier(
        os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    )
    face_embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    rgb_img, faces = FaceDetection.detect(
        "./dataset/pins_Brie Larson/Brie Larson7_1085.jpg", face_detector
    )
    FaceDetection.draw(label="Adriana", rgb_img=rgb_img, faces=faces)
    embeddings = FaceEmbedding.embedding(rgb_img, faces, face_embedder)


def abc():
    img = image.load_img('resources/mini_test/alex_lawther/detected_face_16920.jpg', target_size=(224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    saved_model = load_model("resources/vgg16_1.h5")
    output = saved_model.predict(img)
    print("output", output)


if __name__ == "__main__":
    # train_data, test_data = get_dataset(
    #     "./resources/mini_faces", "./resources/mini_test"
    # )
    # folder_count, file_count = count_file_and_folder("./resources/mini_faces")
    # print("folder_count", folder_count, file_count)
    # print(train_data.num_classes, train_data.class_indices)
    # print(test_data.num_classes, test_data.class_indices)

    abc()
