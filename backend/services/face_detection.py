import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from PIL import Image


class FaceDetection:
    @staticmethod
    def detect(img_path: str, face_detector: MTCNN):
        image = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_faces(rgb_img)
        return rgb_img, faces

    @staticmethod
    def cascade_detect(img_path: str, face_detector: cv2.CascadeClassifier):
        image = cv2.imread(img_path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray_img, scaleFactor=1.3, minNeighbors=5
        )
        return image, faces

    @staticmethod
    def draw(
        label: str,
        rgb_img: cv2.UMat,
        faces: list,
        display_confidence=False,
        display_landmark=False,
    ):
        for i, face in enumerate(faces):
            x, y, w, h = face["box"]
            cv2.rectangle(
                rgb_img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2
            )
            if display_confidence:
                confidence = str(face["confidence"])
                if y > 20:
                    cv2.putText(
                        rgb_img,
                        confidence,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )
                else:
                    cv2.putText(
                        rgb_img,
                        confidence,
                        (x, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )
            if display_landmark:
                for key, value in face["keypoints"].items():
                    plt.scatter(value[0], value[1], s=30, color="blue", marker="o")
                    plt.text(value[0] + 5, value[1], key, color="blue")
        plt.imshow(Image.fromarray(rgb_img))
        plt.title(f"face: {label}")
        plt.axis("off")
        plt.show()

    @staticmethod
    def cascade_draw(label: str, rgb_img: cv2.UMat, faces: list):
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(
                rgb_img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2
            )
        plt.imshow(Image.fromarray(rgb_img))
        plt.title(f"face: {label}")
        plt.axis("off")
        plt.show()
