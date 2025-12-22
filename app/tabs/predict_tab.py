import os
from unittest import result
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QFileDialog, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from core.predict import predict

class DragDropLabel(QLabel):
    def __init__(self, parent_tab):
        super().__init__()
        self.parent_tab = parent_tab
        self.setObjectName("DragDrop") 
        self.setAlignment(Qt.AlignCenter)
        self.setText("<b>Drag Image Here</b><br><span style='font-size:12px; color:#888'>or use the Browse button</span>")
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400) 

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.parent_tab.load_image(file_path)

class PredictTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(30)

        
        left_panel = QVBoxLayout()
        left_panel.setAlignment(Qt.AlignTop)

       
        title = QLabel("Analysis Control")
        title.setObjectName("Header")
        left_panel.addWidget(title)

        
        input_frame = QFrame()
        input_frame.setObjectName("Panel")
        input_layout = QVBoxLayout(input_frame)
        
        input_layout.addWidget(QLabel("File Path:"))
        
        path_row = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Paste path or drag image...")
        self.path_input.returnPressed.connect(lambda: self.load_image(self.path_input.text()))
        
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(40)
        browse_btn.clicked.connect(self.browse_file)
        
        path_row.addWidget(self.path_input)
        path_row.addWidget(browse_btn)
        input_layout.addLayout(path_row)
        
        left_panel.addWidget(input_frame)
        left_panel.addSpacing(20)

        
        self.predict_btn = QPushButton("Run Analysis")
        self.predict_btn.setObjectName("ActionButton") 
        self.predict_btn.setCursor(Qt.PointingHandCursor)
        self.predict_btn.clicked.connect(self.run_prediction)
        left_panel.addWidget(self.predict_btn)

        
        left_panel.addSpacing(20)
        left_panel.addWidget(QLabel("Result:"))
        
        self.result_label = QLabel("Waiting for input...")
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        
        self.result_label.setStyleSheet("color: #ccc; font-size: 14px;") 
        
        left_panel.addWidget(self.result_label)
        left_panel.addStretch()

        
        right_panel = QVBoxLayout()
        
        preview_label = QLabel("Preview")
        preview_label.setStyleSheet("font-weight:bold; margin-bottom: 5px;")
        right_panel.addWidget(preview_label)

        self.image_drop_area = DragDropLabel(self)
        right_panel.addWidget(self.image_drop_area)

        
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

        self.setLayout(main_layout)

    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if fname:
            self.load_image(fname)

    def load_image(self, path):
        path = path.strip('"').strip("'")
        if os.path.exists(path):
            self.path_input.setText(path)
            pixmap = QPixmap(path)
            if pixmap.isNull(): return

            w, h = self.image_drop_area.width(), self.image_drop_area.height()
            scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_drop_area.setPixmap(scaled)
            self.result_label.setText("Ready to analyze.")
        else:
            self.result_label.setText("Error: File not found.")

    def run_prediction(self):
        path = self.path_input.text()
        if not path or not os.path.exists(path):
            self.result_label.setText("Please select a valid image.")
            return

        try:
            self.result_label.setText("Analyzing...")
            QApplication.processEvents()
            
            result = predict(path)

            cls = result["class"]
            conf = result["confidence"]
            text = result["text"]
            
            conf_color = "#27ae60" if conf > 0.7 else "#e67e22" if conf > 0.4 else "#c0392b"
            
            html = (
                f"<h2 style='margin:0; color:{conf_color}'>{cls.upper()}</h2>"
                f"<div style='font-size:14px; color:#aaa; margin-bottom:10px;'>Confidence: {conf:.1%}</div>"
                f"<p style='margin-top:5px; font-style:italic;'>{text}</p>"
            )
            self.result_label.setText(html)
            
        except Exception as e:
            self.result_label.setText(f"Error: {e}")

from PySide6.QtWidgets import QApplication