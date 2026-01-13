from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QTextEdit, QVBoxLayout, QWidget, QProgressBar)

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

        # Progress 
        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def start_training(self):
        epochs = self.epochs_spin.value()
        self.log_box.clear()
        self.log_box.append(f"Starting training session ({epochs} epochs)...")
        self.train_btn.setEnabled(False)

        self.worker = TrainWorker(epochs)
        self.worker.log.connect(self.log_box.append)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, info):
        # compute overall percent across epochs
        if not isinstance(info, dict):
            return
        epoch = info.get("epoch", 1)
        batch = info.get("batch")
        total_batches = info.get("total_batches")
        num_epochs = self.epochs_spin.value()

        if batch and total_batches:
            percent = int(100 * (((epoch - 1) + (batch / total_batches)) / max(1, num_epochs)))
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"Epoch {epoch}/{num_epochs} — Batch {batch}/{total_batches} — {percent}%")
            # loss in the  log
            loss = info.get("loss")
            if loss is not None:
                self.log_box.append(f"Batch {batch}/{total_batches} — loss: {loss:.4f}")
        else:
            phase = info.get("phase")
            if phase == "epoch_summary":
                self.log_box.append(f"Epoch {info.get('epoch')}: Train {info.get('train_acc'):.1%} | Val {info.get('val_acc'):.1%}")
            elif phase == "finished":
                best = info.get("best_val_acc")
                if best is not None:
                    self.log_box.append(f"Training finished. Best Val Acc: {best:.1%}")
                self.progress_bar.setValue(100)

    def on_finished(self):
        self.train_btn.setEnabled(True)
        self.log_box.append(">>> Training session finished.")
