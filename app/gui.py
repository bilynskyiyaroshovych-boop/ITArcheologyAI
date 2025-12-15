import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QTabWidget,
    QComboBox,
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

    # ================= DOWNLOADER TAB =================
    def downloader_tab(self):
        w = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Dataset downloader")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # ---- dataset size selector ----
        size_layout = QHBoxLayout()
        size_label = QLabel("Images per category:")

        self.limit_combo = QComboBox()
        self.limit_combo.addItems(["100", "300", "500", "1000"])
        self.limit_combo.setCurrentIndex(0)  # початкове значення 100

        size_layout.addWidget(size_label)
        size_layout.addWidget(self.limit_combo)
        size_layout.addStretch()
        layout.addLayout(size_layout)

        # ---- start button ----
        self.download_btn = QPushButton("Start download")
        self.download_btn.clicked.connect(self.start_download)
        layout.addWidget(self.download_btn)

        # ---- log output ----
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Logs will appear here...")
        layout.addWidget(self.log_box)

        w.setLayout(layout)
        return w

    def start_download(self):
        limit = int(self.limit_combo.currentText())

        tasks = [
            ("ancient pottery", "ceramics", limit),
            ("bronze age jewelry", "jewelry", limit),
            ("neolithic tools", "tools", limit),
            ("archaeological pottery fragments", "fragments", limit),
            ("ancient beads", "beads", limit),
        ]

        self.download_btn.setEnabled(False)
        self.log_box.append(
            f"Starting download: {limit} images per category..."
        )

        self.worker = DownloadWorker(tasks)
        self.worker.log.connect(self.log_box.append)
        self.worker.finished.connect(self.download_finished)
        self.worker.start()

    def download_finished(self):
        self.log_box.append("All downloads finished.")
        self.download_btn.setEnabled(True)

    # ================= TRAINING TAB =================
    def training_tab(self):
        w = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Training (coming next)"))
        layout.addWidget(QPushButton("Start training"))

        w.setLayout(layout)
        return w

    # ================= PREDICT TAB =================
    def predict_tab(self):
        w = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Prediction (coming next)"))
        layout.addWidget(QPushButton("Choose image"))

        w.setLayout(layout)
        return w


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
