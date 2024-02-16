from PyQt6 import QtWidgets
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent
from PyQt6.QtWidgets import QLabel, QWidget, QPushButton
from mtcnn import MTCNN

from services.app_service import analytic_img
from services.face_detection import FaceDetection
from utils import to_fixed

BASE_WIDTH = 600


class Qt6App(QWidget):
    def __init__(self):
        self.face_locations = []
        self.face_detector = MTCNN()

        super().__init__()
        self.setWindowTitle("Face recognition")
        self.resize(BASE_WIDTH, 700)

        self.location_label = QLabel("location: ---", self)
        self.location_label.resize(150, 35)
        self.location_label.setGeometry(10, 40, BASE_WIDTH, 35)

        self.size_label = QLabel("size: ---", self)
        self.size_label.resize(150, 35)
        self.size_label.setGeometry(10, 80, BASE_WIDTH, 35)

        self.name_label = QLabel("name: ---", self)
        self.name_label.resize(150, 35)
        self.name_label.setGeometry(10, 120, BASE_WIDTH, 35)

        self.label = QLabel(self)
        self.label.resize(BASE_WIDTH, self.height() - 160)
        self.label.setGeometry(0, 160, 600, self.height() - 160)

        choose_btn = QPushButton("Choose image", self)
        choose_btn.setCheckable(True)
        choose_btn.move(10, 0)
        choose_btn.resize(100, 35)
        choose_btn.clicked.connect(self.add_image)

        quick_btn = QPushButton("Quit", self)
        quick_btn.move(130, 0)
        quick_btn.resize(100, 35)
        quick_btn.clicked.connect(QCoreApplication.instance().quit)

    def mousePressEvent(self, event: QMouseEvent):
        x = event.pos().x()
        y = event.pos().y()
        for location in self.face_locations:
            x_location = location['x']
            y_location = location['y'] + 160
            w = location['w']
            h = location['h']
            if x_location <= x <= x_location + w and y_location <= y <= y_location + h:
                self.location_label.setText(f"location: x: {to_fixed(x_location)}, y: {to_fixed(y_location)}")
                self.size_label.setText(f"size: w: {to_fixed(w)}, h: {to_fixed(h)}")
                self.name_label.setText(f"name: x: {to_fixed(x_location)}, y: {to_fixed(y_location)}")

    def _resize_img(self, _path: str):
        _, faces = FaceDetection.detect(_path, self.face_detector)
        image, face_locations, new_height = analytic_img(_path, faces, BASE_WIDTH)
        self.face_locations = face_locations
        self.label.resize(BASE_WIDTH, new_height)
        _convert = QImage(
            image,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            QImage.Format.Format_BGR888,
        )
        self.label.setPixmap(QPixmap.fromImage(_convert))

    def add_image(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select image(s)", "", "Images (*.png *.jpg *.jpeg)"
        )
        if len(paths) > 0:
            self._resize_img(paths[0])
