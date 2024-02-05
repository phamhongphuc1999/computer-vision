import cv2
import os
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from mtcnn import MTCNN

from services.face_detection import FaceDetection
from services.face_embedding import FaceEmbedding
from services.face_recognition import FaceRecognition
from services.preprocessing import Preprocessing

# if __name__ == "__main__":
#     face_detector = MTCNN()
#     face_cascade = cv2.CascadeClassifier(
#         os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
#     )
#     face_embedder = InceptionResnetV1(pretrained="vggface2").eval()
#
#     rgb_img, faces = FaceDetection.detect("./images/mask-person.jpg", face_detector)
#     rgb_img1, faces1 = FaceDetection.detect(
#         "./dataset/pins_Adriana Lima/Adriana Lima1_1.jpg", face_detector
#     )
#     rgb_img1, faces1 = FaceDetection.detect(
#         "./dataset/pins_Chris Pratt/Chris Pratt0_722.jpg", face_detector
#     )
#     FaceDetection.draw(
#         label="Adriana",
#         rgb_img=rgb_img,
#         faces=faces,
#     )
#     FaceDetection.draw(
#         label="Adriana1",
#         rgb_img=rgb_img1,
#         faces=faces1,
#         display_confidence=True,
#         display_landmark=True,
#     )
#     embeddings1 = FaceEmbedding.embedding(rgb_img, faces, face_embedder)
#     embeddings2 = FaceEmbedding.embedding(rgb_img1, faces1, face_embedder)
#     distance = FaceRecognition.embedding_distance(embeddings1[0], embeddings2[0])
#     print(f"distance: {distance:.4f}")
#
#     image, faces = FaceDetection.cascade_detect(
#         "./images/mask-person.jpg", face_cascade
#     )
#     FaceDetection.cascade_draw("Adriana", cv2.cvtColor(image, cv2.COLOR_BGR2RGB), faces)

if __name__ == "__main__":
    image_paths = Preprocessing.get_image_paths("./dataset")
    print("image_paths", len(image_paths), image_paths[0])
