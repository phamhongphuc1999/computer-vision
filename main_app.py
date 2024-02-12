import sys

import cv2
from PyQt6 import QtWidgets
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QLabel, QApplication, QWidget, QPushButton
from mtcnn import MTCNN
from services.face_detection import FaceDetection


class Window(QWidget):
    def __init__(self):
        self.face_detector = MTCNN()

        super().__init__()
        self.setWindowTitle("Face recognition")

        self.label = QLabel(self)
        self.label.resize(500, 500)
        self.label.setContentsMargins(0, 50, 0, 0)

        button = QPushButton("CLICK", self)
        button.setCheckable(True)
        button.resize(150, 50)
        button.clicked.connect(self.add_image)

    def add_image(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select image(s)", "", "Images (*.png *.jpg *.jpeg)"
        )
        if len(paths) > 0:
            _path = paths[0]
            pixmap = QPixmap(paths[0])
            pixmap = pixmap.scaled(450, 450)
            self.label.setPixmap(pixmap)

            rgb_img, faces = FaceDetection.detect(_path, self.face_detector)
            for i, face in enumerate(faces):
                x, y, w, h = face["box"]
                cv2.rectangle(
                    rgb_img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2
                )
            _convert = QImage(
                rgb_img,
                rgb_img.shape[1],
                rgb_img.shape[0],
                rgb_img.strides[0],
                QImage.Format.Format_BGR888,
            )
            self.label.setPixmap(QPixmap.fromImage(_convert))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
