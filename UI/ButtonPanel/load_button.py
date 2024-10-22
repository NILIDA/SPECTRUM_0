from PyQt6.QtWidgets import QPushButton, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QVBoxLayout, QGroupBox, QFormLayout, QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from .load_image.edit_img import ImageEditWindow


class LoadButton(QPushButton):
    def __init__(self, spectrum_data_frame):
        super().__init__('Загрузить .txt')
        self.spectrum_data_frame = spectrum_data_frame
        self.clicked.connect(self.on_click)

    def on_click(self):
        self.spectrum_data_frame.load_spectrum_txt()

class LoadImg(QPushButton):
    def __init__(self, spectrum_data_frame):
        super().__init__('Загрузить изображение по умолчанию')
        self.spectrum_data_frame = spectrum_data_frame
        self.clicked.connect(self.on_click)

    def on_click(self):
        self.spectrum_data_frame.load_image()


class LoadExtImg(QPushButton):
    def __init__(self, spectrum_data_frame):
        super().__init__('Подбор параметров')
        self.clicked.connect(self.on_click)
    def on_click(self):
        self.imageProcessor = ImageEditWindow()
        self.imageProcessor.show()
        self.image = None