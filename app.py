from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QPushButton, QColorDialog, QFileDialog, QWidget, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import cv2
from image import sharpen, make
from cp.onnx_model import face_parser

class PhotoApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Photo Viewer with Color Adjustment")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        # Left menu layout
        self.menu_layout = QVBoxLayout()
        self.main_layout.addLayout(self.menu_layout, stretch=1)

        # Create a scroll area for parts
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)
        self.menu_layout.addWidget(self.scroll_area)

        # Attributes and colors table
        self.table = {
            'face': 1,
            'left_brow': 2,
            'right_brow': 3,
            'left_eye': 4,
            'right_eye': 5,
            'glasses': 6,
            'left_ear': 7,
            'right_ear': 8,
            'nose': 10,
            'mouth': 11,
            'upper_lip': 12,
            'lower_lip': 13,
            'neck': 14,
            'neck_l': 15,
            'cloth': 16,
            'hair': 17,
            'hat': 18
        }

        self.part_colors = {key: None for key in self.table}
        self.color_buttons = {}

        # Create color pickers for each part
        for part_name in self.table.keys():
            part_label = QLabel(part_name.capitalize())
            self.scroll_layout.addWidget(part_label)

            color_button = QPushButton("Pick Color")
            color_button.clicked.connect(lambda _, p=part_name: self.pick_part_color(p))
            self.scroll_layout.addWidget(color_button)

            self.color_buttons[part_name] = color_button

        # Buttons
        self.load_image_button = QPushButton("Load Center Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.menu_layout.addWidget(self.load_image_button)

        self.apply_filter_button = QPushButton("Apply")
        self.apply_filter_button.clicked.connect(self.apply_make)
        self.menu_layout.addWidget(self.apply_filter_button)

        self.save_filter_button = QPushButton("Save")
        self.save_filter_button.clicked.connect(self.save_image)
        self.menu_layout.addWidget(self.save_filter_button)

        # Image layout
        self.image_layout = QHBoxLayout()
        self.main_layout.addLayout(self.image_layout, stretch=3)

        # Labels for images
        self.image1_label = QLabel()
        self.image2_label = QLabel()

        self.image1_label.setAlignment(Qt.AlignCenter)
        self.image2_label.setAlignment(Qt.AlignCenter)

        self.image_layout.addWidget(self.image1_label)
        self.image_layout.addWidget(self.image2_label)

        # Placeholder for images
        self.original_image = None
        self.filtered_image = None

    def pick_part_color(self, part_name):
        color = QColorDialog.getColor()
        if color.isValid():
            self.part_colors[part_name] = [color.blue(), color.green(), color.red()]
            self.color_buttons[part_name].setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                self.display_image(self.original_image, self.image1_label)
            except Exception as e:
                print(f"Error loading image: {e}")

    def save_image(self):
        if self.filtered_image is not None:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
            if file_path:
                try:
                    cv2.imwrite(file_path, self.filtered_image)
                except Exception as e:
                    print(f"Error saving image: {e}")

    def apply_make(self):
        if self.original_image is not None:
            image = self.original_image.copy()

            parsing = face_parser(image)
            parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

            for part_name, part_id in self.table.items():
                color = self.part_colors[part_name]
                if color is not None:
                    image = make(image, parsing, part_id, color)

            self.filtered_image = image
            self.display_image(self.filtered_image, self.image2_label)



    def display_image(self, image, label):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = PhotoApp()
    window.show()
    sys.exit(app.exec_())
