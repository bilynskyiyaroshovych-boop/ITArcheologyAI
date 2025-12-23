import traceback

from PySide6.QtCore import QThread, Signal

from core.downloader import run_downloader
from core.train_model import train


class DownloadWorker(QThread):
    log = Signal(str)
    finished = Signal()

    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks

    def run(self):
        self.log.emit("Starting download process...")
        try:
            run_downloader(self.tasks)
            self.log.emit("Download finished successfully.")
        except Exception as e:
            self.log.emit(f"Error: {e}")
        self.finished.emit()


class TrainWorker(QThread):
    log = Signal(str)
    finished = Signal()

    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def run(self):
        self.log.emit("Initializing training...")
        try:

            train(num_epochs=self.epochs, log_callback=self.log.emit)
        except Exception:

            self.log.emit(f"Critical Error:\n{traceback.format_exc()}")
        self.finished.emit()
