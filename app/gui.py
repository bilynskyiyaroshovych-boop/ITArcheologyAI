import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QLabel,
    QTabWidget, QTextEdit
)
from app.workers import DownloadWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Archaeological Artifact Classifier")
        self.resize(900, 600)

        self.worker = None

        tabs = QTabWidget()
        tabs.addTab(self.downloader_tab(), "Downloader")
        tabs.addTab(self.training_tab(), "Training")
        tabs.addTab(self.predict_tab(), "Predict")

        self.setCentralWidget(tabs)

    # ---------- DOWNLOADER TAB ----------
    def downloader_tab(self):
        w = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Dataset downloader"))

        self.download_btn = QPushButton("Start download")
        self.download_btn.clicked.connect(self.start_download)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        layout.addWidget(self.download_btn)
        layout.addWidget(self.log_box)

        w.setLayout(layout)
        return w

    def start_download(self):
        tasks = [
            ("ancient pottery", "ceramics", 100),
            ("bronze age jewelry", "jewelry", 100),
            ("neolithic tools", "tools", 100),
            ("archaeological pottery fragments", "fragments", 100),
            ("ancient beads", "beads", 100),
        ]

        self.download_btn.setEnabled(False)
        self.log_box.append("Preparing tasks...")

        self.worker = DownloadWorker(tasks)
        self.worker.log.connect(self.log_box.append)
        self.worker.finished.connect(self.download_finished)
        self.worker.start()

    def download_finished(self):
        self.log_box.append("All downloads done.")
        self.download_btn.setEnabled(True)

    # ---------- TRAINING TAB ----------
    def training_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Training (next step)"))
        layout.addWidget(QPushButton("Start training"))
        w.setLayout(layout)
        return w

    # ---------- PREDICT TAB ----------
    def predict_tab(self):
        w = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Prediction (next step)"))
        layout.addWidget(QPushButton("Choose image"))
        w.setLayout(layout)
        return w


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
