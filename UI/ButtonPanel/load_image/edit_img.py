import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSlider,
    QFileDialog, QSplitter, QFrame, QSpinBox
)
from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image, ImageFilter, ImageQt
import numpy as np
import cv2

class ImageEditWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(QSize(920, 600))
        self.setMaximumSize(QSize(1200, 700))
        self.initUI()
        self.image = None

    def initUI(self):
        self.setWindowTitle('Редактирование изображения')

        self.main_layout = QHBoxLayout()

        # Create splitter to separate parameter panel and image display area
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create parameter panel
        self.param_panel = QFrame(self)
        self.param_layout = QVBoxLayout(self.param_panel)

        self.load_button = QPushButton('Загрузить изображение')
        self.load_button.clicked.connect(self.loadImage)
        self.param_layout.addWidget(self.load_button)

        self.blur_label = QLabel('Размытие:')
        self.param_layout.addWidget(self.blur_label)
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(1, 20)
        self.blur_slider.setValue(5)
        self.blur_slider.valueChanged.connect(self.updateImage)
        self.param_layout.addWidget(self.blur_slider)

        self.block_size_label = QLabel('Размер блока:')
        self.param_layout.addWidget(self.block_size_label)
        self.block_size_slider =  QSlider(Qt.Orientation.Horizontal)
        self.block_size_slider.setRange(1, 100)
        self.block_size_slider.setValue(17)
        self.block_size_slider.valueChanged.connect(self.updateImage)
        self.param_layout.addWidget(self.block_size_slider)

        self.apply_button = QPushButton('Продолжить')
        self.apply_button.clicked.connect(self.applyChanges)
        self.param_layout.addWidget(self.apply_button)

        self.param_panel.setLayout(self.param_layout)
        self.param_panel.setFixedWidth(200)  # Set fixed width for the parameter panel
        self.splitter.addWidget(self.param_panel)

        self.image_label = QLabel('Здесь появится изображение')
        self.image_label.setFixedSize(800, 600)  # Set fixed size for the image display area
        self.splitter.addWidget(self.image_label)

        self.main_layout.addWidget(self.splitter)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        self.max_width = 800
        self.max_height = 600


    def loadImage(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg)")
        if file_dialog.exec():
            file_names = file_dialog.selectedFiles()
            if file_names:
                fileName = file_names[0]
                try:
                    self.image = Image.open(fileName).convert('RGBA')
                    self.updateImage()
                except Exception as e:
                    print(f"Error loading image: {e}")

    def updateImage(self):
        if self.image:
            try:
                blur = self.blur_slider.value()
                block_size = self.block_size_slider.value()
                processed_image = self.crop(self.image, blur, block_size)
                qimage = self.pil2qimage(processed_image)
                pixmap = QPixmap.fromImage(qimage)

                # Масштабирование изображения с учетом пропорций
                scaled_pixmap = pixmap.scaled(self.max_width, self.max_height, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)

                self.image_label.setPixmap(scaled_pixmap)
                print("Image updated successfully")
            except Exception as e:
                print(f"Error updating image: {e}")

    def applyChanges(self):
        if self.image:
            blur = self.blur_slider.value()
            block_size = self.block_size_slider.value()
            processed_image = self.crop(self.image, blur, block_size)
            self.nextWindow = ColorEditWindow(processed_image)
            self.nextWindow.show()

    def crop(self, img, blur, block_size):
        try:
            # Преобразование изображения в градации серого
            gray = img.convert('L')
            gray_np = np.array(gray)

            # Применение размытия Гаусса
            if blur % 2 == 0:
                blur += 1  # Гауссово размытие требует нечетного размера ядра
            blured = cv2.GaussianBlur(gray_np, (blur, blur), 0)

            if block_size % 2 == 0:
                block_size += 1

            # Функция адаптивного порогового фильтра Берсена
            def bersen_thresholding(image, block_size, c):
                binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                     block_size, c)
                return binary_image

            # Применение адаптивного порогового фильтра Берсена
            binary_bersen = bersen_thresholding(blured, block_size, 15)

            # Алгоритм Собеля
            sobel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])

            gradient_x = cv2.filter2D(binary_bersen, cv2.CV_64F, sobel_x)
            gradient_y = cv2.filter2D(binary_bersen, cv2.CV_64F, sobel_y)

            gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

            # Бинаризация
            threshold = np.mean(gradient)
            binary_gradient = np.where(gradient > threshold, 255, 0).astype(np.uint8)  # Преобразование типа данных

            # Поиск контуров
            contours, _ = cv2.findContours(binary_gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Нахождение прямоугольника с максимальной площадью
            max_area = 0
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            # Выравнивание и вырезание области
            rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Вычисление матрицы поворота
            angle = rect[-1]
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90

            M = cv2.getRotationMatrix2D((rect[0][0], rect[0][1]), angle, 1.0)

            # Поворот изображения
            img_np = np.array(img)  # Преобразование PIL изображения в массив NumPy
            rotated = cv2.warpAffine(img_np, M, (img_np.shape[1], img_np.shape[0]))

            # Вырезание области
            x, y, w, h = cv2.boundingRect(max_contour)
            cropped = rotated[y:y + h, x:x + w]

            return Image.fromarray(cropped)
        except Exception as e:
            print(f"Error in crop function: {e}")
            return img

    def pil2qimage(self, image):
        try:
            data = image.tobytes("raw", "RGBA")
            qimage = QImage(data, image.size[0], image.size[1], QImage.Format.Format_RGBA8888)
            return qimage
        except Exception as e:
            print(f"Error converting PIL image to QImage: {e}")
            return QImage()


class ColorEditWindow(QMainWindow):
    graph_result = QtCore.pyqtSignal(object)
    def __init__(self, img):
        super().__init__()
        self.setMinimumSize(QSize(920, 600))
        self.setMaximumSize(QSize(1200, 700))
        self.image = img
        self.initUI(img)

    def initUI(self, img):
        self.setWindowTitle('Поиск графика')
        self.main_layout = QHBoxLayout()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.param_panel = QFrame(self)
        self.param_layout = QVBoxLayout(self.param_panel)

        self.l_b_label = QLabel('Минимальное значение синего:')
        self.param_layout.addWidget(self.l_b_label)
        self.l_b = QSlider(Qt.Orientation.Horizontal)
        self.l_b.setRange(0, 225)
        self.l_b.setValue(100)
        self.l_b.valueChanged.connect(self.updateImage)
        self.param_layout.addWidget(self.l_b)

        self.u_b_label = QLabel('Максимальное значение синего:')
        self.param_layout.addWidget(self.u_b_label)
        self.u_b = QSlider(Qt.Orientation.Horizontal)
        self.u_b.setRange(0, 225)
        self.u_b.setValue(220)
        self.u_b.valueChanged.connect(self.updateImage)
        self.param_layout.addWidget(self.u_b)

        self.l_g_label = QLabel('Минимальное значение зеленого:')
        self.param_layout.addWidget(self.l_g_label)
        self.l_g = QSlider(Qt.Orientation.Horizontal)
        self.l_g.setRange(0, 225)
        self.l_g.setValue(100)
        self.l_g.valueChanged.connect(self.updateImage)
        self.param_layout.addWidget(self.l_g)

        self.u_g_label = QLabel('Максимальное значение зеленого:')
        self.param_layout.addWidget(self.u_g_label)
        self.u_g = QSlider(Qt.Orientation.Horizontal)
        self.u_g.setRange(0, 225)
        self.u_g.setValue(220)
        self.u_g.valueChanged.connect(self.updateImage)
        self.param_layout.addWidget(self.u_g)

        self.l_r_label = QLabel('Минимальное значение красного:')
        self.param_layout.addWidget(self.l_r_label)
        self.l_r = QSlider(Qt.Orientation.Horizontal)
        self.l_r.setRange(0, 225)
        self.l_r.setValue(100)
        self.l_r.valueChanged.connect(self.updateImage)
        self.param_layout.addWidget(self.l_r)

        self.u_r_label = QLabel('Максимальное значение красного:')
        self.param_layout.addWidget(self.u_r_label)
        self.u_r = QSlider(Qt.Orientation.Horizontal)
        self.u_r.setRange(0, 225)
        self.u_r.setValue(220)
        self.u_r.valueChanged.connect(self.updateImage)
        self.param_layout.addWidget(self.u_r)

        self.apply_button = QPushButton('Продолжить')
        self.apply_button.clicked.connect(self.applyChanges)
        self.param_layout.addWidget(self.apply_button)

        self.param_panel.setLayout(self.param_layout)
        self.param_panel.setFixedWidth(200)
        self.splitter.addWidget(self.param_panel)

        self.image_label = QLabel('Здесь появится изображение')
        self.image_label.setFixedSize(800, 600)  # Set fixed size for the image display area
        self.splitter.addWidget(self.image_label)
        qimage = self.pil2qimage(img)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))
        self.main_layout.addWidget(self.image_label)

        self.main_layout.addWidget(self.splitter)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        self.max_width = 700
        self.max_height = 600

    def pil2qimage(self, image):
        try:
            if image.mode == '1':
                image = image.convert('L')
            if image.mode == 'L':
                image = image.convert('RGBA')
            qimage = ImageQt.ImageQt(image)
            return qimage
        except Exception as e:
            print(f"Error converting PIL image to QImage: {e}")
            return QImage()

    def updateImage(self):
        if self.image:
            try:
                l_b = self.l_b.value()
                l_g = self.l_g.value()
                l_r = self.l_r.value()
                u_b = self.u_b.value()
                u_g = self.u_g.value()
                u_r = self.u_r.value()
                processed_image = self.cut(self.image, l_b, l_g, l_r, u_b, u_g, u_r)
                qimage = self.pil2qimage(processed_image)
                pixmap = QPixmap.fromImage(qimage)

                # Масштабирование изображения с учетом пропорций
                scaled_pixmap = pixmap.scaled(self.max_width, self.max_height,
                                              aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio)

                self.image_label.setPixmap(scaled_pixmap)
                print("Image updated successfully")
            except Exception as e:
                print(f"Error updating image: {e}")


    def cut(self, img, l_b, l_g, l_r, u_b, u_g, u_r):
        lower_blue = np.array([l_b, l_g, l_r])
        upper_blue = np.array([u_b, u_g, u_r])

        # Преобразование изображения в массив NumPy
        img_np = np.array(img)

        # Преобразование изображения в цветовое пространство BGR
        image_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Создание маски, соответствующей области темно-синего цвета
        mask = cv2.inRange(image_bgr, lower_blue, upper_blue)

        # Применение маски к исходному изображению
        result = cv2.bitwise_and(img_np, img_np, mask=mask)

        # Преобразование результата в оттенки серого
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Бинаризация: все пиксели, которые не являются черными (0), становятся белыми (255)
        _, binary_image = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Инверсия бинарного изображения, чтобы белые пиксели стали черными, а черные остались без изменений
        binary_image = cv2.bitwise_not(binary_image)

        # Преобразование бинарного изображения в объект PIL Image
        binary_image_pil = Image.fromarray(binary_image)

        return binary_image_pil

    def applyChanges(self):
        l_b = self.l_b.value()
        l_g = self.l_g.value()
        l_r = self.l_r.value()
        u_b = self.u_b.value()
        u_g = self.u_g.value()
        u_r = self.u_r.value()
        result = self.cut(self.image, l_b, l_g, l_r, u_b, u_g, u_r)
        # self.graph_result.emit(result)

if __name__ == '__main__':
    app = QApplication(sys.argv)