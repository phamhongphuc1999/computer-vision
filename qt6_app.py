from utils import to_fixed
from PyQt6 import QtWidgets
from PyQt6.QtCore import QCoreApplication
from services.app_service import AppService
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent
from PyQt6.QtWidgets import QLabel, QWidget, QPushButton

BASE_WIDTH = 500


class Qt6App(QWidget):
    def __init__(self):
        self._path = ""
        self._show_mark = False
        self.face_locations = []
        self.app_service = AppService()

        super().__init__()
        self.setWindowTitle("Face recognition")
        self.resize(BASE_WIDTH, 600)

        self.path_label = QLabel("path: ---", self)
        self.path_label.resize(150, 25)
        self.path_label.setGeometry(10, 40, BASE_WIDTH, 35)

        self.location_label = QLabel("location: ---", self)
        self.location_label.resize(150, 25)
        self.location_label.setGeometry(10, 70, BASE_WIDTH, 35)

        self.size_label = QLabel("size: ---", self)
        self.size_label.resize(150, 25)
        self.size_label.setGeometry(10, 100, BASE_WIDTH, 35)

        self.name_label = QLabel("name: ---", self)
        self.name_label.resize(150, 25)
        self.name_label.setGeometry(10, 130, BASE_WIDTH, 35)

        self.label = QLabel(self)
        self.label.resize(BASE_WIDTH, self.height() - 160)
        self.label.setGeometry(0, 160, BASE_WIDTH, self.height() - 160)

        choose_btn = QPushButton("Choose image", self)
        choose_btn.setCheckable(True)
        choose_btn.move(10, 0)
        choose_btn.resize(100, 35)
        choose_btn.clicked.connect(self._import_image)

        self.mark_btn = QPushButton("Show", self)
        self.mark_btn.setCheckable(True)
        self.mark_btn.move(120, 0)
        self.mark_btn.resize(100, 35)
        self.mark_btn.clicked.connect(self._mark_click)

        quick_btn = QPushButton("Quit", self)
        quick_btn.move(230, 0)
        quick_btn.resize(100, 35)
        quick_btn.clicked.connect(QCoreApplication.instance().quit)

    def mousePressEvent(self, event: QMouseEvent):
        x = event.pos().x()
        y = event.pos().y()
        for location in self.face_locations:
            x_location = location["x"]
            y_location = location["y"] + 160
            w = location["w"]
            h = location["h"]
            if x_location <= x <= x_location + w and y_location <= y <= y_location + h:
                self.location_label.setText(
                    f"location: x: {to_fixed(x_location)}, y: {to_fixed(y_location)}"
                )
                self.size_label.setText(f"size: w: {to_fixed(w)}, h: {to_fixed(h)}")
                self.name_label.setText(
                    f"name: {location['predicted_name']}({location['percent']})"
                )

    def _resize_img(self, _path: str, show_mark=False):
        image, face_locations, new_height = self.app_service.analytic(
            _path, BASE_WIDTH, show_mark=show_mark
        )
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
        self.resize(BASE_WIDTH, new_height + 160)

    def _import_image(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select image(s)", "", "Images (*.png *.jpg *.jpeg)"
        )
        if len(paths) > 0:
            self._path = paths[0]
            self.path_label.setText(f"path: {paths[0][0:5]}...{paths[0][-40:-1]}")
            self._resize_img(paths[0])

    def _mark_click(self):
        if self._show_mark:
            self._show_mark = False
            self.mark_btn.setText("Hide")
            self._resize_img(self._path, False)
        else:
            self._show_mark = True
            self.mark_btn.setText("Show")
            self._resize_img(self._path, True)
