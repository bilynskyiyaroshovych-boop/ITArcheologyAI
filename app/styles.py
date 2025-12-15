# app/styles.py


PRO_THEME = """
/* ================= Settings ================= */
QWidget {
    background-color: #2b2b2b;      
    color: #e0e0e0;                 
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;                
}

/* ================= (Header) ================= */
QLabel#Header {
    font-size: 20px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 10px;
}

/* ================= (Panel) ================= */
/* Рамки навколо кнопок налаштувань */
QFrame#Panel {
    background-color: #363636;
    border: 1px solid #444;
    border-radius: 8px;
}

/* ================= (ActionButton) ================= */
/* Великі кнопки: Start Download, Train, Analyze */
QPushButton#ActionButton {
    background-color: #0078d4;      
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-size: 15px;
    font-weight: bold;
    min-width: 150px;               
}

QPushButton#ActionButton:hover {
    background-color: #106ebe;
}

QPushButton#ActionButton:pressed {
    background-color: #005a9e;
}

QPushButton#ActionButton:disabled {
    background-color: #444;
    color: #888;
}

/* ================= Buttons ================= */
/* Маленькі кнопки типу "Browse..." */
QPushButton {
    background-color: #444;
    border: 1px solid #555;
    color: white;
    border-radius: 4px;
    padding: 5px 10px;
}
QPushButton:hover {
    background-color: #505050;
}

/* ================= (Inputs) ================= */
QLineEdit, QSpinBox, QComboBox {
    background-color: #1e1e1e;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px;
    color: white;
    selection-background-color: #0078d4;
}

/* ================= Logs ================= */
QTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #333;
    color: #00ff00;                 
    font-family: 'Consolas', monospace;
    font-size: 13px;
}

/* ================= DRAG & DROP ================= */
QLabel#DragDrop {
    border: 2px dashed #666;
    background-color: #252525;
    color: #aaa;
    border-radius: 12px;
}
QLabel#DragDrop:hover {
    border-color: #0078d4;
    background-color: #2a2d30;
    color: white;
}

/* ================= (Tabs) ================= */
QTabWidget::pane {
    border: 1px solid #444;
    background: #2b2b2b;
    top: -1px; 
}
QTabBar::tab {
    background: #2b2b2b;
    color: #aaa;
    padding: 8px 25px;
    border: 1px solid transparent;
    border-bottom: 2px solid transparent; 
}
QTabBar::tab:selected {
    color: white;
    font-weight: bold;
    border-bottom: 2px solid #0078d4; 
}
QTabBar::tab:hover {
    background-color: #333;
}
"""