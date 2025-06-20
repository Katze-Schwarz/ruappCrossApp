import sys
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSpinBox, QFileDialog, QMessageBox, QRadioButton, 
                             QLineEdit, QListWidget, QScrollArea, QSizePolicy, QFrame, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import (QPixmap, QImage, QFont, QColor, QPainter, QPen, QIcon, 
                         QAction, QGuiApplication)
from dmc_palette import DMC_COLORS

# Constants
SYMBOL_BORDER_SIZE = 0
IMAGE_SIZE = 100
MIN_SIZE = 50
MAX_SIZE = 500
SCALE_FACTOR = 40
DPI = 300
PDF_SIZE = (11.69, 8.27)
SYMBOL_FONT_SCALE = 0.6
SYMBOL_FONT_THICKNESS = 1
SYMBOL_BORDER_COLOR = (0, 0, 0)
MAX_PREVIEW_SIZE = 600
CACHE_SIZE = 3
DEFAULT_IMAGE_SIZE = 100
RESIZE_DELAY = 1000

COLORS = {
    'primary': '#E1BEE7',  # Very light purple
    'primary_light': '#F3E5F5',  # Almost white purple
    'secondary': '#CE93D8',  # Light purple
    'accent': '#BA68C8',  # Medium purple (accent)
    'background': '#FAF5FF',  # Very light background
    'text': '#000000',  # Black text
    'text_light': "#000000",  # Black text (same)
    'border': '#D1C4E9',  # Borders
    'success': '#AB47BC',  # Success (purple)
    'warning': '#9C27B0',  # Warning (purple)
    'danger': '#8E24AA'  # Error (purple)
}

# Application styles
APP_STYLE = f"""
    QMainWindow {{
        background-color: {COLORS['background']};
    }}
    
    QMessageBox {{
        background-color: white;
    }}
    QMessageBox QLabel {{
        color: {COLORS['text']};
    }}
    QMessageBox QPushButton {{
        background-color: {COLORS['primary']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 5px 10px;
        min-width: 70px;
    }}
    QMessageBox QPushButton:hover {{
        background-color: {COLORS['secondary']};
    }}
    
    QGroupBox {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        margin-top: 10px;
        padding-top: 15px;
        font-weight: bold;
        color: {COLORS['text']};
        background-color: {COLORS['primary_light']};
    }}
    
    QPushButton {{
        background-color: {COLORS['primary']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 8px 12px;
        font-weight: bold;
        min-width: 80px;
    }}
    
    QPushButton:hover {{
        background-color: {COLORS['secondary']};
    }}
    
    QPushButton:pressed {{
        background-color: {COLORS['accent']};
        color: white;
    }}
    
    QSpinBox, QLineEdit, QListWidget {{
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 5px;
        background-color: white;
        color: {COLORS['text']};
    }}
    
    QLabel {{
        color: {COLORS['text']};
        background-color: transparent;
    }}
    
    QScrollArea {{
        background-color: {COLORS['primary_light']};
        border: none;
    }}
    
    QScrollBar:vertical {{
        width: 12px;
        background: {COLORS['primary_light']};
    }}
    
    QScrollBar::handle:vertical {{
        background: {COLORS['secondary']};
        min-height: 20px;
        border-radius: 6px;
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        background: none;
    }}
"""

class ImageCache:
    def __init__(self, size=CACHE_SIZE):
        self.cache = {}
        self.order = []
        self.size = size

    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache.get(key)
        return None

    def set(self, key, value):
        if key not in self.cache:
            if len(self.order) >= self.size:
                oldest = self.order.pop(0)
                del self.cache[oldest]
            self.cache[key] = value
            self.order.append(key)

class PreviewThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            result = self.generate_preview_task(self.params)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def generate_preview_task(self, params):
        image_path, scheme_width, num_colors, palette_var, custom_dmc, aspect_ratio = params
        
        scheme_height = int(scheme_width / aspect_ratio)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_small = cv2.resize(img_rgb, (scheme_width, scheme_height), interpolation=cv2.INTER_AREA)
        pixels = img_small.reshape(-1, 3).astype(np.float32) / 255.0
        
        selected_palette = DMC_COLORS if palette_var == "base" else custom_dmc
        n_clusters = min(len(selected_palette), num_colors)
        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(pixels)
        clustered = (kmeans.cluster_centers_ * 255).astype(np.uint8)
        clustered_img = clustered[labels].reshape(img_small.shape)
        
        used_colors = np.unique(clustered_img.reshape(-1, 3), axis=0)
        dmc_lab = prepare_dmc_colors(selected_palette)
        
        symbols = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]
        color_info = []
        used_codes = set()
        
        for i, color in enumerate(used_colors):
            result = find_nearest_dmc(color, dmc_lab, used_codes)
            if result:
                code, name, rgb = result
                used_codes.add(code)
                color_info.append({
                    'symbol': symbols[i % len(symbols)],
                    'code': code,
                    'name': name,
                    'rgb': rgb,
                    'count': np.sum(labels == i)
                })
                
        large_img = cv2.resize(
            clustered_img,
            (img_small.shape[1] * SCALE_FACTOR, img_small.shape[0] * SCALE_FACTOR),
            interpolation=cv2.INTER_NEAREST
        )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for y in range(clustered_img.shape[0]):
            for x in range(clustered_img.shape[1]):
                color = clustered_img[y, x]
                idx = np.where(np.all(used_colors == color, axis=1))[0][0]
                symbol = color_info[idx]['symbol'] if idx < len(color_info) else "?"
                pos_x = x * SCALE_FACTOR + SCALE_FACTOR // 2
                pos_y = y * SCALE_FACTOR + SCALE_FACTOR // 2
                brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
                cv2.putText(large_img, symbol, (pos_x, pos_y),
                            font, SYMBOL_FONT_SCALE, SYMBOL_BORDER_COLOR,
                            SYMBOL_FONT_THICKNESS + SYMBOL_BORDER_SIZE, cv2.LINE_AA)
                cv2.putText(large_img, symbol, (pos_x, pos_y),
                            font, SYMBOL_FONT_SCALE, text_color,
                            SYMBOL_FONT_THICKNESS, cv2.LINE_AA)
                            
        grid_color = (150, 150, 150)
        for y in range(0, large_img.shape[0], SCALE_FACTOR):
            cv2.line(large_img, (0, y), (large_img.shape[1], y), grid_color, 1)
        for x in range(0, large_img.shape[1], SCALE_FACTOR):
            cv2.line(large_img, (x, 0), (x, large_img.shape[0]), grid_color, 1)
            
        return large_img

def prepare_dmc_colors(dmc_list):
    if not hasattr(prepare_dmc_colors, 'cache'):
        prepare_dmc_colors.cache = {}
        
    cache_key = tuple((code, r, g, b) for code, name, r, g, b in dmc_list)
    if cache_key in prepare_dmc_colors.cache:
        return prepare_dmc_colors.cache[cache_key]
    
    dmc_lab = []
    for code, name, r, g, b in dmc_list:
        rgb = np.array([r/255, g/255, b/255], dtype=np.float32).reshape(1, 1, 3)
        lab = rgb2lab(rgb)[0, 0]
        dmc_lab.append((code, name, (r, g, b), lab))
    
    prepare_dmc_colors.cache[cache_key] = dmc_lab
    return dmc_lab

def find_nearest_dmc(target_rgb, dmc_lab, used_codes=None):
    if used_codes is None:
        used_codes = set()
    rgb = np.array(target_rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    target_lab = rgb2lab(rgb)[0, 0]
    min_distance = float('inf')
    nearest_dmc = None
    for code, name, rgb_val, lab_val in dmc_lab:
        if code in used_codes:
            continue
        distance = deltaE_ciede2000(target_lab, lab_val)
        if distance < min_distance:
            min_distance = distance
            nearest_dmc = (code, name, rgb_val)
    return nearest_dmc

class CrossStitchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cross Stitch Pattern Generator")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(APP_STYLE)
        
        self.image_path = ""
        self.custom_dmc = []
        self.dmc_dict = {code: (name, (r, g, b)) for code, name, r, g, b in DMC_COLORS}
        self.current_image = None
        self.original_size = (0, 0)
        self.aspect_ratio = 1.0
        self.image_cache = ImageCache()
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.generate_preview)
        self.preview_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # ===================== LEFT PANEL =====================
        left_panel_container = QWidget()
        left_panel_container.setFixedWidth(450)
        left_panel_container.setStyleSheet(f"""
            background-color: {COLORS['primary_light']};
            border-radius: 10px;
            padding: 5px;
        """)
        
        left_panel_layout = QVBoxLayout(left_panel_container)
        left_panel_layout.setContentsMargins(5, 5, 5, 5)
        left_panel_layout.setSpacing(10)
        
        # Header with icon
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        icon_label = QLabel()
        icon_pixmap = QPixmap(":icon.png").scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        icon_label.setPixmap(icon_pixmap)
        
        title = QLabel("Cross Stitch Generator")
        title.setStyleSheet(f"""
            QLabel {{
                font-size: 20px;
                font-weight: bold;
                color: {COLORS['text']};
            }}
        """)
        
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        left_panel_layout.addWidget(title_container)
        
        # Styled separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"color: {COLORS['border']};")
        left_panel_layout.addWidget(separator)
        
        # Scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        # Main content of left panel
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(15)
        
        # Image selection group
        image_group = QGroupBox("Image Selection")
        image_group.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {COLORS['border']};
            }}
        """)
        image_layout = QVBoxLayout()
        
        self.img_btn = QPushButton("Select Image")
        self.img_btn.setIcon(QIcon.fromTheme("document-open"))
        self.img_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                padding: 10px;
                font-size: 14px;
                color: {COLORS['text']};
            }}
        """)
        
        self.img_path_label = QLabel("No file selected")
        self.img_path_label.setWordWrap(True)
        self.img_path_label.setStyleSheet(f"""
            QLabel {{
                background-color: white;
                padding: 10px;
                border-radius: 6px;
                border: 1px solid {COLORS['border']};
                min-height: 40px;
                color: {COLORS['text']};
            }}
        """)
        
        image_layout.addWidget(self.img_btn)
        image_layout.addWidget(self.img_path_label)
        image_group.setLayout(image_layout)
        content_layout.addWidget(image_group)
        
        # Pattern parameters group
        params_group = QGroupBox("Pattern Parameters")
        params_layout = QVBoxLayout()
        
        # Pattern size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Width:"))
        self.scheme_width = QSpinBox()
        self.scheme_width.setRange(MIN_SIZE, MAX_SIZE)
        size_layout.addWidget(self.scheme_width)
        
        size_layout.addWidget(QLabel("Height:"))
        self.scheme_height_label = QLabel()
        self.scheme_height_label.setStyleSheet("font-weight: bold;")
        size_layout.addWidget(self.scheme_height_label)
        size_layout.addStretch()
        
        params_layout.addLayout(size_layout)
        
        # Color count
        colors_layout = QHBoxLayout()
        colors_layout.addWidget(QLabel("Colors:"))
        self.num_colors = QSpinBox()
        self.num_colors.setRange(1, 100)
        colors_layout.addWidget(self.num_colors)
        
        self.auto_colors_btn = QPushButton("Auto Pick")
        self.auto_colors_btn.setIcon(QIcon.fromTheme("color-picker"))
        colors_layout.addWidget(self.auto_colors_btn)
        
        params_layout.addLayout(colors_layout)
        
        # Palette selection
        palette_layout = QHBoxLayout()
        self.palette_base = QRadioButton("DMC")
        self.palette_custom = QRadioButton("Custom")
        self.palette_base.setChecked(True)
        
        palette_layout.addWidget(self.palette_base)
        palette_layout.addWidget(self.palette_custom)
        palette_layout.addStretch()
        
        params_layout.addLayout(palette_layout)
        params_group.setLayout(params_layout)
        content_layout.addWidget(params_group)
        
        # Color palette group
        palette_group = QGroupBox("Color Palette")
        palette_group_layout = QVBoxLayout()
        
        # Input field with buttons
        dmc_input_layout = QHBoxLayout()
        self.dmc_entry = QLineEdit()
        self.dmc_entry.setPlaceholderText("DMC color code")
        dmc_input_layout.addWidget(self.dmc_entry)
        
        self.add_btn = QPushButton("+")
        self.add_btn.setFixedWidth(40)
        self.remove_btn = QPushButton("-")
        self.remove_btn.setFixedWidth(40)
        
        dmc_input_layout.addWidget(self.add_btn)
        dmc_input_layout.addWidget(self.remove_btn)
        palette_group_layout.addLayout(dmc_input_layout)
        
        # Color list with improved display
        self.dmc_listbox = QListWidget()
        self.dmc_listbox.setStyleSheet(f"""
            QListWidget {{
                background-color: white;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {COLORS['accent']};
                color: {COLORS['text']};
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['secondary']};
                color: {COLORS['text']};
            }}
        """)
        palette_group_layout.addWidget(self.dmc_listbox)
        
        # Palette management buttons
        palette_buttons_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.save_btn = QPushButton("Save")
        self.load_btn = QPushButton("Load")
        
        palette_buttons_layout.addWidget(self.clear_btn)
        palette_buttons_layout.addWidget(self.save_btn)
        palette_buttons_layout.addWidget(self.load_btn)
        palette_group_layout.addLayout(palette_buttons_layout)
        
        palette_group.setLayout(palette_group_layout)
        content_layout.addWidget(palette_group)
        
        # Generation buttons
        buttons_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setIcon(QIcon.fromTheme("document-preview"))
        self.generate_btn = QPushButton("Generate Pattern")
        self.generate_btn.setIcon(QIcon.fromTheme("document-save"))
        
        self.preview_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                padding: 12px;
                font-size: 14px;
                color: {COLORS['text']};
            }}
        """)
        
        self.generate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                padding: 12px;
                font-size: 14px;
                color: white;
            }}
        """)
        
        buttons_layout.addWidget(self.preview_btn)
        buttons_layout.addWidget(self.generate_btn)
        content_layout.addLayout(buttons_layout)
        
        scroll.setWidget(content_widget)
        left_panel_layout.addWidget(scroll)
        
        # ===================== RIGHT PANEL =====================
        right_panel = QWidget()
        right_panel.setStyleSheet(f"""
            background-color: white;
            border-radius: 10px;
        """)
        
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        
        # Preview header
        preview_header = QHBoxLayout()
        preview_title = QLabel("Pattern Preview")
        preview_title.setStyleSheet(f"""
            QLabel {{
                font-size: 18px;
                font-weight: bold;
                color: {COLORS['text']};
            }}
        """)
        
        self.loading_label = QLabel()
        self.loading_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['danger']};
                font-weight: bold;
            }}
        """)
        
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        preview_header.addWidget(self.loading_label)
        
        right_layout.addLayout(preview_header)
        
        # Preview area
        self.preview_container = QWidget()
        self.preview_container.setStyleSheet(f"""
            background-color: {COLORS['primary_light']};
            border-radius: 8px;
        """)
        
        preview_container_layout = QVBoxLayout(self.preview_container)
        preview_container_layout.setContentsMargins(5, 5, 5, 5)
        
        self.preview_image = QLabel()
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setStyleSheet("background-color: white; border-radius: 6px;")
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.preview_image)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
            }
        """)
        
        preview_container_layout.addWidget(scroll_area)
        right_layout.addWidget(self.preview_container)
        
        # Pattern info
        self.preview_label = QLabel("Image not loaded")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(f"""
            QLabel {{
                background-color: white;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid {COLORS['border']};
                font-style: italic;
                color: {COLORS['text']};
            }}
        """)
        right_layout.addWidget(self.preview_label)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel_container)
        main_layout.addWidget(right_panel)
        
        # Set stretch
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 2)
        
        # Connect signals
        self.img_btn.clicked.connect(self.select_image)
        self.auto_colors_btn.clicked.connect(self.autopick_color_count)
        self.add_btn.clicked.connect(self.add_dmc_color)
        self.remove_btn.clicked.connect(self.remove_dmc_color)
        self.clear_btn.clicked.connect(self.clear_dmc_colors)
        self.save_btn.clicked.connect(self.save_custom_palette)
        self.load_btn.clicked.connect(self.load_custom_palette)
        self.preview_btn.clicked.connect(self.generate_preview)
        self.generate_btn.clicked.connect(self.generate_pattern)
        self.scheme_width.valueChanged.connect(self.update_height_from_width)

    def update_height_from_width(self):
        try:
            width = self.scheme_width.value()
            height = int(width / self.aspect_ratio)
            self.scheme_height_label.setText(str(height))
        except ZeroDivisionError:
            self.scheme_height_label.setText("Calculation error")
            
    def autopick_color_count(self):
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
    
        try:
            scheme_width = self.scheme_width.value()
            scheme_height = int(scheme_width / self.aspect_ratio)
            img = cv2.imread(self.image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_small = cv2.resize(img_rgb, (scheme_width, scheme_height), interpolation=cv2.INTER_AREA)
        
            img_lab = cv2.cvtColor(img_small, cv2.COLOR_RGB2LAB)
            unique_colors = np.unique(img_lab.reshape(-1, 3), axis=0)
        
            def is_similar(color1, color2, threshold=10):
                return deltaE_ciede2000(color1, color2) < threshold
        
            filtered_colors = []
            for color in unique_colors:
                 if not any(is_similar(color, f) for f in filtered_colors):
                     filtered_colors.append(color)
        
            num_unique = len(filtered_colors)
            selected_palette = DMC_COLORS if self.palette_base.isChecked() else self.custom_dmc
            max_in_palette = len(selected_palette)
        
            if num_unique <= 10:
                optimal_colors = num_unique
            elif num_unique <= 20:
               optimal_colors = min(num_unique + 2, max_in_palette)
            else:
               complexity_factor = 0.25
               optimal_colors = min(
                   int(num_unique * complexity_factor) + 8,
                   max_in_palette,
                   40
                )
        
            self.num_colors.setValue(optimal_colors)
            self.generate_preview()
        
            QMessageBox.information(self, "Auto Colors", 
                          f"Recommended color count: {optimal_colors}\n"
                          f"Unique colors in image: {num_unique}\n"
                          f"Available in palette: {max_in_palette}")
    
        except Exception as e:
           QMessageBox.critical(self, "Error", f"Failed to determine color count:\n{str(e)}")

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", 
                                           "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.image_path = path
            self.img_path_label.setText(os.path.basename(path))
            self.loading_label.setText("Loading...")
            QApplication.processEvents()
            
            try:
                img = cv2.imread(path)
                if img is None:
                    QMessageBox.critical(self, "Error", "Failed to load image")
                    return
                h, w = img.shape[:2]
                self.original_height, self.original_width = h, w
                self.aspect_ratio = w / h
                self.scheme_width.setValue(min(IMAGE_SIZE, self.original_width))
                self.update_height_from_width()
                self.generate_preview()
            finally:
                self.loading_label.setText("")

    def generate_preview(self):
        if not self.image_path:
            return
            
        current_params = (
            self.image_path,
            self.scheme_width.value(),
            self.num_colors.value(),
            "base" if self.palette_base.isChecked() else "custom",
            tuple(self.custom_dmc),
            self.aspect_ratio
        )
        
        if hasattr(self, '_last_preview_params') and self._last_preview_params == current_params:
            return
        self._last_preview_params = current_params
        
        self.loading_label.setText("Generating...")
        QApplication.processEvents()
        
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.terminate()
            
        self.preview_thread = PreviewThread(current_params)
        self.preview_thread.finished.connect(self.preview_generated)
        self.preview_thread.error.connect(self.handle_preview_error)
        self.preview_thread.start()

    def preview_generated(self, image):
        self.loading_label.setText("")
        self.update_preview_display(image)

    def update_preview_display(self, image):
        try:
            self.current_image = image.copy()
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            height, width, channel = img_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale pixmap to fit preview area
            preview_size = self.preview_image.size()
            scaled_pixmap = pixmap.scaled(preview_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            self.preview_image.setPixmap(scaled_pixmap)
            self.preview_label.setText(
                f"Pattern size: {image.shape[1]//SCALE_FACTOR}x{image.shape[0]//SCALE_FACTOR} stitches"
            )
            
        except Exception as e:
            self.handle_preview_error(e)

    def handle_preview_error(self, error):
        self.loading_label.setText("")
        self.preview_image.clear()
        self.preview_label.setText("Preview generation error")
        QMessageBox.critical(self, "Error", f"Failed to generate preview:\n{str(error)}")

    def add_dmc_color(self):
        code = self.dmc_entry.text().strip().upper()
        if not code:
            return
        if code in self.dmc_dict:
            name, (r, g, b) = self.dmc_dict[code]
            if code not in {c[0] for c in self.custom_dmc}:
                self.custom_dmc.append((code, name, r, g, b))
                self.dmc_listbox.addItem(f"{code} - {name}")
                self.dmc_entry.clear()
            else:
                QMessageBox.warning(self, "Warning", "Color already added")
        else:
            QMessageBox.critical(self, "Error", f"Color '{code}' not found in database")

    def remove_dmc_color(self):
        selected = self.dmc_listbox.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Warning", "Select a color to remove.")
            return
            
        removed = self.dmc_listbox.item(selected).text()
        code = removed.split(" - ")[0]
        self.custom_dmc = [color for color in self.custom_dmc if color[0] != code]
        self.dmc_listbox.takeItem(selected)

    def clear_dmc_colors(self):
        reply = QMessageBox.question(self, "Confirmation", "Are you sure?", 
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.custom_dmc.clear()
            self.dmc_listbox.clear()
            self.generate_preview()
            QMessageBox.information(self, "Information", "Palette cleared.")

    def save_custom_palette(self):
        if not self.custom_dmc:
            QMessageBox.warning(self, "Warning", "No colors to save.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Palette", "", 
                                            "CSV files (*.csv)")
        if path:
            df = pd.DataFrame(self.custom_dmc, columns=["code", "name", "r", "g", "b"])
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Success", f"Palette saved to:\n{path}")

    def load_custom_palette(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Palette", "", 
                                            "CSV files (*.csv)")
        if path:
            try:
                df = pd.read_csv(path)
                loaded = [(str(row['code']), str(row['name']), int(row['r']), int(row['g']), int(row['b'])) 
                         for _, row in df.iterrows()]
                self.custom_dmc = loaded
                self.dmc_listbox.clear()
                for code, name, r, g, b in self.custom_dmc:
                    self.dmc_listbox.addItem(f"{code} - {name}")
                self.generate_preview()
                QMessageBox.information(self, "Success", f"Loaded {len(loaded)} colors.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load palette:\n{str(e)}")

    def generate_pattern(self):
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return
            
        self.loading_label.setText("Generating pattern...")
        QApplication.processEvents()
        
        try:
            save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
            if not save_dir:
                self.loading_label.setText("")
                return
                
            scheme_width = self.scheme_width.value()
            scheme_height = int(scheme_width / self.aspect_ratio)
            img = cv2.imread(self.image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_small = cv2.resize(img_rgb, (scheme_width, scheme_height), interpolation=cv2.INTER_AREA)
            pixels = img_small.reshape(-1, 3).astype(np.float32) / 255.0
            
            selected_palette = DMC_COLORS if self.palette_base.isChecked() else self.custom_dmc
            n_clusters = min(len(selected_palette), self.num_colors.value())
            
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
            labels = kmeans.fit_predict(pixels)
            clustered = (kmeans.cluster_centers_ * 255).astype(np.uint8)
            clustered_img = clustered[labels].reshape(img_small.shape)
            
            used_colors = np.unique(clustered_img.reshape(-1, 3), axis=0)
            dmc_lab = prepare_dmc_colors(selected_palette)
            
            symbols = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]
            color_info = []
            used_codes = set()
            
            for i, color in enumerate(used_colors):
                result = find_nearest_dmc(color, dmc_lab, used_codes)
                if result:
                    code, name, rgb = result
                    used_codes.add(code)
                    color_info.append({
                        'symbol': symbols[i % len(symbols)],
                        'code': code,
                        'name': name,
                        'rgb': rgb,
                        'count': np.sum(labels == i)
                    })
                    
            large_img = cv2.resize(
                clustered_img,
                (img_small.shape[1] * SCALE_FACTOR, img_small.shape[0] * SCALE_FACTOR),
                interpolation=cv2.INTER_NEAREST
            )
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            for y in range(clustered_img.shape[0]):
                for x in range(clustered_img.shape[1]):
                    color = clustered_img[y, x]
                    idx = np.where(np.all(used_colors == color, axis=1))[0][0]
                    symbol = color_info[idx]['symbol'] if idx < len(color_info) else "?"
                    pos_x = x * SCALE_FACTOR + SCALE_FACTOR // 2
                    pos_y = y * SCALE_FACTOR + SCALE_FACTOR // 2
                    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                    text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
                    cv2.putText(large_img, symbol, (pos_x, pos_y),
                                font, SYMBOL_FONT_SCALE, SYMBOL_BORDER_COLOR,
                                SYMBOL_FONT_THICKNESS + SYMBOL_BORDER_SIZE, cv2.LINE_AA)
                    cv2.putText(large_img, symbol, (pos_x, pos_y),
                                font, SYMBOL_FONT_SCALE, text_color,
                                SYMBOL_FONT_THICKNESS, cv2.LINE_AA)
            
            grid_color = (150, 150, 150)
            for y in range(0, large_img.shape[0], SCALE_FACTOR):
                cv2.line(large_img, (0, y), (large_img.shape[1], y), grid_color, 1)
            for x in range(0, large_img.shape[1], SCALE_FACTOR):
                cv2.line(large_img, (x, 0), (x, large_img.shape[0]), grid_color, 1)
                
            output_img = os.path.join(save_dir, "cross_stitch_pattern.png")
            output_pdf = os.path.join(save_dir, "cross_stitch_pattern.pdf")
            cv2.imwrite(output_img, cv2.cvtColor(large_img, cv2.COLOR_RGB2BGR))
            self.create_pdf(color_info, output_pdf, output_img)
            
            QMessageBox.information(self, "Success", f"Files saved to:\n{save_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
        finally:
            self.loading_label.setText("")

    def create_pdf(self, color_info, pdf_path, image_path):
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=PDF_SIZE, dpi=DPI)
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.suptitle("Cross Stitch Pattern", fontsize=12)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            fig = plt.figure(figsize=PDF_SIZE, dpi=DPI)
            ax = fig.add_subplot(111)
            ax.axis('off')
            sorted_info = sorted(color_info, key=lambda x: x['count'], reverse=True)
            cell_data = [["Symbol", "DMC Code", "Color Name", "Stitch Count"]]
            cell_colors = [['#f0f0f0'] * 4]
            for info in sorted_info:
                r, g, b = info['rgb']
                cell_data.append([
                    info['symbol'],
                    info['code'],
                    info['name'],
                    str(info['count'])
                ])
                cell_colors.append([(r/255, g/255, b/255)] * 4)
            table = ax.table(
                cellText=cell_data,
                cellColours=cell_colors,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            for i in range(len(cell_data)):
                for j in range(4):
                    if i == 0:
                        table[(i, j)].set_text_props(weight='bold', color='black')
                    else:
                        r, g, b = sorted_info[i-1]['rgb']
                        luminance = 0.299*r + 0.587*g + 0.114*b
                        text_color = 'black' if luminance > 150 else 'white'
                        table[(i, j)].set_text_props(color=text_color)
            plt.suptitle("Color Legend", fontsize=12)
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CrossStitchApp()
    window.show()
    sys.exit(app.exec())