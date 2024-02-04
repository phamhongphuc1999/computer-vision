import numpy
from sklearn.metrics.pairwise import euclidean_distances


class FaceRecognition:
    @staticmethod
    def embedding_distance(embedding1: numpy.ndarray, embedding2: numpy.ndarray):
        return euclidean_distances(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
