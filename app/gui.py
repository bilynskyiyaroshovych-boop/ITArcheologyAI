import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from app.styles import PRO_THEME
from app.tabs.downloader_tab import DownloaderTab
from app.tabs.predict_tab import PredictTab
from app.tabs.train_model_tab import TrainingTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Archaeological Artifact Classifier")
        self.resize(1000, 750)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(DownloaderTab(), " Downloader")
        self.tabs.addTab(TrainingTab(), " Training")
        self.tabs.addTab(PredictTab(), " Predict")


def run_gui():
    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    app.setStyleSheet(PRO_THEME)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
