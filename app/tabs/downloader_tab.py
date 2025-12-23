from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QTextEdit, QVBoxLayout, QWidget)

from app.workers import DownloadWorker


class DownloaderTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        header = QLabel("Dataset Management")
        header.setObjectName("Header")
        layout.addWidget(header)

        control_frame = QFrame()
        control_frame.setObjectName("Panel")

        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(15)

        lbl = QLabel("Images per category:")

        self.limit_combo = QComboBox()
        self.limit_combo.addItems(["100", "300", "600", "1000"])
        self.limit_combo.setFixedWidth(120)

        self.download_btn = QPushButton("Start Download")
        self.download_btn.setObjectName("ActionButton")
        self.download_btn.setCursor(Qt.PointingHandCursor)
        self.download_btn.clicked.connect(self.start_download)

        control_layout.addWidget(lbl)
        control_layout.addWidget(self.limit_combo)
        control_layout.addStretch()
        control_layout.addWidget(self.download_btn)

        layout.addWidget(control_frame)

        layout.addWidget(QLabel("Process Logs:"))

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def start_download(self):
        limit = int(self.limit_combo.currentText())
        tasks = [
            ("ancient pottery", "ceramics", limit),
            ("bronze age jewelry", "jewelry", limit),
            ("neolithic tools", "tools", limit),
            ("archaeological pottery fragments", "fragments", limit),
            ("ancient beads", "beads", limit),
        ]
        self.log_box.clear()
        self.log_box.append(f"Initializing download ({limit} items)...")
        self.download_btn.setEnabled(False)

        self.worker = DownloadWorker(tasks)
        self.worker.log.connect(self.log_box.append)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self):
        self.log_box.append(">>> All tasks completed.")
        self.download_btn.setEnabled(True)
