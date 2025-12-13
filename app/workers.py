from PySide6.QtCore import QThread, Signal
from core.downloader import run_downloader


class DownloadWorker(QThread):
    log = Signal(str)
    finished = Signal()

    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks

    def run(self):
        self.log.emit("Starting download...")
        try:
            run_downloader(self.tasks)
            self.log.emit("Download finished successfully.")
        except Exception as e:
            self.log.emit(f"Error: {e}")
        self.finished.emit()
