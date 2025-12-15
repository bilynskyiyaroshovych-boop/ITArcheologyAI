from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QSpinBox, QFrame
)
from PySide6.QtCore import Qt
from app.workers import TrainWorker

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        
        header = QLabel("Model Training")
        header.setObjectName("Header")
        layout.addWidget(header)

        
        control_frame = QFrame()
        control_frame.setObjectName("Panel")
        
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(15)
        
        lbl = QLabel("Training Epochs:")
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10)
        self.epochs_spin.setValue(5)
        self.epochs_spin.setFixedWidth(80)
        
        
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setObjectName("ActionButton")
        self.train_btn.setCursor(Qt.PointingHandCursor)
        self.train_btn.clicked.connect(self.start_training)

        control_layout.addWidget(lbl)
        control_layout.addWidget(self.epochs_spin)
        control_layout.addStretch()
        control_layout.addWidget(self.train_btn)

        layout.addWidget(control_frame)

        
        layout.addWidget(QLabel("Training Output:"))
        
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def start_training(self):
        epochs = self.epochs_spin.value()
        self.log_box.clear()
        self.log_box.append(f"Starting training session ({epochs} epochs)...")
        self.train_btn.setEnabled(False)

        self.worker = TrainWorker(epochs)
        self.worker.log.connect(self.log_box.append)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self):
        self.train_btn.setEnabled(True)
        self.log_box.append(">>> Training session finished.")