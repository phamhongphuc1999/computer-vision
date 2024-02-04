import cv2
import torch
from facenet_pytorch import InceptionResnetV1, extract_face


class FaceEmbedding:
    @staticmethod
    def embedding(rgb_img: cv2.UMat, faces: list, face_embedder: InceptionResnetV1):
        embeddings = [face_embedder(extract_face(rgb_img, face['box']).unsqueeze(0)).squeeze().detach().numpy()
                      for face in faces]
        return embeddings
