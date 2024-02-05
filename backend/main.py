import cv2
import os

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from mtcnn import MTCNN

from services.face_detection import FaceDetection
from services.face_embedding import FaceEmbedding

if __name__ == "__main__":
    device = "cpu"
    face_detector = MTCNN()
    face_cascade = cv2.CascadeClassifier(
        os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    )
    face_embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    rgb_img, faces = FaceDetection.detect("./images/mask-person.jpg", face_detector)
    FaceDetection.draw(label="Adriana", rgb_img=rgb_img, faces=faces)
    embeddings = FaceEmbedding.embedding(rgb_img, faces, face_embedder)
