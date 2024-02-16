import sys
from PyQt6.QtWidgets import QApplication
from qt6_app import Qt6App


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Qt6App()
    window.show()
    sys.exit(app.exec())
