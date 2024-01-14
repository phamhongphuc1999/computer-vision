import pickle
from typing import Sequence

import cv2
import face_recognition

from app.services import convert_xywh_to_recognition, draw_xywh_image, get_img
from app.services.face_detection import FaceDetection


class FaceRecognition:
    @staticmethod
    def recognition(encoding_path: str, image_path: str = None, image=None):
        data = pickle.loads(open(encoding_path, "rb").read())
        real_image = get_img(image_path=image_path, image=image)
        rgb = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        faces = FaceDetection.face_locations(image=real_image)
        face_locations = map(convert_xywh_to_recognition, faces)

        encodings = face_recognition.face_encodings(
            face_image=rgb, known_face_locations=face_locations
        )
        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            if True in matches:
                matched_index_list = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matched_index_list:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
            names.append(name)
        return faces, names

    @staticmethod
    def draw(image, faces: Sequence[int], names: Sequence[str]):
        for (location, name) in zip(faces, names):
            if name != 'Unknown':
                draw_xywh_image(image, location, name)
