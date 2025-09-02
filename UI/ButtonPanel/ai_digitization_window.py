# UI/ButtonPanel/ai_digitization_window.py
import cv2
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QInputDialog, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QWidget
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from logger_config import logger
from tensorflow.keras.models import load_model

class AxisInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ввод значений осей")
        self.layout = QVBoxLayout()
        self.x_min_input = QLineEdit()
        self.x_max_input = QLineEdit()
        self.y_min_input = QLineEdit()
        self.y_max_input = QLineEdit()
        self.layout.addWidget(QLabel("Минимальное значение X (длина волны):"))
        self.layout.addWidget(self.x_min_input)
        self.layout.addWidget(QLabel("Максимальное значение X (длина волны):"))
        self.layout.addWidget(self.x_max_input)
        self.layout.addWidget(QLabel("Минимальное значение Y (интенсивность):"))
        self.layout.addWidget(self.y_min_input)
        self.layout.addWidget(QLabel("Максимальное значение Y (интенсивность):"))
        self.layout.addWidget(self.y_max_input)
        self.confirm_button = QPushButton("Подтвердить")
        self.confirm_button.clicked.connect(self.accept)
        self.layout.addWidget(self.confirm_button)
        self.setLayout(self.layout)

    def get_inputs(self):
        try:
            x_min = float(self.x_min_input.text()) if self.x_min_input.text() else 0
            x_max = float(self.x_max_input.text()) if self.x_max_input.text() else 4000
            y_min = float(self.y_min_input.text()) if self.y_min_input.text() else -10
            y_max = float(self.y_max_input.text()) if self.y_max_input.text() else 100
            return x_min, x_max, y_min, y_max
        except ValueError:
            logger.error("Ошибка: Введите числовые значения для осей")
            return 0, 4000, -10, 100

class AIDigitizationWindow(QMainWindow):
    def __init__(self, spectrum_data_frame):
        super().__init__()
        self.spectrum_data_frame = spectrum_data_frame
        self.setWindowTitle("Оцифровка графика с помощью ИИ")
        self.setGeometry(100, 100, 800, 600)
        self.image = None
        self.image_cv = None
        self.selected_color = None
        self.zoom_factor = 1.0

        # Matplotlib canvas для отображения изображения
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Настройка интерфейса
        self.layout = QVBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.load_button = QPushButton("Загрузить изображение")
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        self.color_button = QPushButton("Выбрать цвет линии")
        self.color_button.clicked.connect(self.pick_color)
        self.layout.addWidget(self.color_button)

        self.axis_button = QPushButton("Установить значения осей")
        self.axis_button.clicked.connect(self.set_axis_values)
        self.layout.addWidget(self.axis_button)

        self.digitize_button = QPushButton("Оцифровать график")
        self.digitize_button.clicked.connect(self.digitize_graph)
        self.layout.addWidget(self.digitize_button)

        self.layout.addWidget(self.canvas)

        # Поддержка зума
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Загрузка модели нейронной сети
        try:
            self.model = load_model('graph_line_detector.h5')
            logger.info("Модель нейронной сети успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.model = None

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            try:
                self.image = plt.imread(file_name)
                self.image_cv = cv2.imread(file_name)
                if self.image is None or self.image_cv is None:
                    logger.error(f"Не удалось загрузить изображение: {file_name}")
                    return
                self.ax.imshow(self.image)
                self.canvas.draw()
                logger.info(f"Изображение загружено: {file_name}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке изображения: {e}")

    def on_scroll(self, event):
        if event.inaxes == self.ax:
            if event.button == 'up':
                self.zoom_factor *= 1.1
            elif event.button == 'down':
                self.zoom_factor /= 1.1
            self.update_image()

    def update_image(self):
        if self.image is not None:
            self.ax.clear()
            self.ax.imshow(self.image)
            self.ax.set_xlim([0, self.image.shape[1] / self.zoom_factor])
            self.ax.set_ylim([self.image.shape[0] / self.zoom_factor, 0])
            self.canvas.draw()

    def pick_color(self):
        if self.image_cv is None:
            logger.warning("Сначала загрузите изображение")
            return
        selected_color = [0, 0, 0]
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_color[:] = self.image_cv[y, x][::-1]  # BGR to RGB
                logger.info(f"Выбранный цвет RGB: {selected_color}")
                bgr_color = np.uint8([[self.image_cv[y, x]]])
                hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
                logger.debug(f"Выбранный цвет в HSV (OpenCV): {hsv_color}")
                hsv_standard = [hsv_color[0] * 2, (hsv_color[1] / 255) * 100, (hsv_color[2] / 255) * 100]
                logger.debug(f"Выбранный цвет в стандартном HSV: {hsv_standard}")
                cv2.destroyAllWindows()
        cv2.imshow("Выберите цвет (кликните по линии, затем нажмите любую клавишу)", self.image_cv)
        cv2.setMouseCallback("Выберите цвет (кликните по линии, затем нажмите любую клавишу)", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.selected_color = selected_color

    def set_axis_values(self):
        dialog = AxisInputDialog(self)
        if dialog.exec():
            self.x_min, self.x_max, self.y_min, self.y_max = dialog.get_inputs()
            logger.info(f"Значения осей установлены: X({self.x_min}, {self.x_max}), Y({self.y_min}, {self.y_max})")

    def detect_graph_bounds(self):
        """Автоматическое определение границ графика на изображении."""
        if self.image_cv is None:
            logger.warning("Изображение не загружено")
            return 0, 0, self.image.shape[1], self.image.shape[0]
        gray = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        graph_rect = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0 and area > 1000:
                    max_area = area
                    graph_rect = (x, y, w, h)
        
        if graph_rect is None:
            logger.warning("Не удалось определить границы графика, используются границы изображения")
            return 0, 0, self.image.shape[1], self.image.shape[0]
        
        x, y, w, h = graph_rect
        return x, y, x + w, y + h

    def preprocess_image(self):
        if self.image_cv is None or self.selected_color is None:
            logger.warning("Необходимо загрузить изображение и выбрать цвет")
            return None
        img_hsv = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2HSV)
        bgr_color = np.uint8([[self.selected_color[::-1]]])  # RGB to BGR
        graph_color_hsv = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
        lower_hsv = np.clip(graph_color_hsv - [10, 30, 30], [0, 0, 0], [180, 255, 255])
        upper_hsv = np.clip(graph_color_hsv + [10, 30, 30], [0, 0, 0], [180, 255, 255])
        logger.debug(f"Диапазон HSV: lower={lower_hsv}, upper={upper_hsv}")
        mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        white_pixels = np.sum(mask == 255)
        total_pixels = mask.size
        logger.debug(f"Белых пикселей в маске: {white_pixels} из {total_pixels} ({white_pixels/total_pixels*100:.2f}%)")
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def digitize_graph(self):
        if self.image_cv is None or self.selected_color is None or not hasattr(self, 'x_min'):
            logger.error("Необходимо загрузить изображение, выбрать цвет и установить значения осей")
            return

        # Предобработка изображения
        mask = self.preprocess_image()
        if mask is None:
            return

        # Сохранение отладочной маски
        cv2.imwrite("debug_color_mask.png", mask)
        logger.info("Сохранена отладочная маска: debug_color_mask.png")

        # Извлечение координат белых пикселей из маски
        white_pixels = np.where(mask == 255)
        all_y, all_x = white_pixels[0], white_pixels[1]
        if len(all_x) == 0:
            logger.error("Не найдено белых пикселей в маске. Проверьте цветовую сегментацию.")
            return

        # Корректируем координаты Y (переворачиваем ось Y)
        all_y = self.image_cv.shape[0] - all_y
        logger.debug(f"После инверсии Y: min_y={np.min(all_y)}, max_y={np.max(all_y)}")

        # Усреднение Y для уникальных X
        all_x, all_y = self.average_y_values(all_x, all_y)
        all_x, all_y = np.array(all_x), np.array(all_y)
        logger.debug(f"После усреднения: len(x)={len(all_x)}, min_x={np.min(all_x)}, max_x={np.max(all_x)}, min_y={np.min(all_y)}, max_y={np.max(all_y)}")

        if len(all_x) == 0:
            logger.error("После усреднения данных не осталось точек.")
            return

        # Определение границ графика
        min_x_pix, min_y_pix, max_x_pix, max_y_pix = self.detect_graph_bounds()
        logger.debug(f"Границы графика: min_x_pix={min_x_pix}, max_x_pix={max_x_pix}, min_y_pix={min_y_pix}, max_y_pix={max_y_pix}")

        # Проверка корректности границ
        if max_x_pix <= min_x_pix or max_y_pix <= min_y_pix:
            logger.error("Некорректные границы графика: max_x_pix <= min_x_pix или max_y_pix <= min_y_pix")
            return

        # Преобразование координат в физические значения
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min

        # Масштабируем all_x на основе реальных пиксельных границ графика
        all_x_normalized = (all_x - min_x_pix) / (max_x_pix - min_x_pix)
        transformed_x = self.x_max - all_x_normalized * x_range

        # Масштабируем all_y с учётом инверсии и добавляем смещение -10
        all_y_normalized = (all_y - min_y_pix) / (max_y_pix - min_y_pix)
        transformed_y = self.y_min + all_y_normalized * y_range - 10

        # Ограничение значений
        transformed_x = np.clip(transformed_x, self.x_min, self.x_max)
        transformed_y = np.clip(transformed_y, self.y_min, self.y_max)
        logger.debug(f"После масштабирования и смещения: min_x={np.min(transformed_x)}, max_x={np.max(transformed_x)}, min_y={np.min(transformed_y)}, max_y={np.max(transformed_y)}")

        # Применение фильтра Савицкого-Голея для сглаживания
        if len(transformed_y) >= 11:
            transformed_y = savgol_filter(transformed_y, window_length=11, polyorder=2)
            logger.debug(f"После сглаживания: min_y={np.min(transformed_y)}, max_y={np.max(transformed_y)}")

        # Сортировка по убыванию X (для ИК-спектров)
        sorted_indices = np.argsort(transformed_x)[::-1]
        transformed_x = transformed_x[sorted_indices]
        transformed_y = transformed_y[sorted_indices]

        # Построение графика в Matplotlib
        plt.figure(figsize=(8, 6))
        plt.plot(transformed_x, transformed_y, label="Оцифрованный график", color="blue")
        plt.xlabel("Длина волны")
        plt.ylabel("Интенсивность")
        plt.title("Оцифрованный график")
        plt.grid(True)
        plt.legend()
        plt.xlim(self.x_max, self.x_min)  # Убывающая ось X
        plt.ylim(self.y_min, self.y_max)
        plt.margins(0)
        plt.savefig("digitized_graph.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        logger.info("График построен и сохранён как digitized_graph.png")

        # Создание DataFrame
        df = pd.DataFrame({"Длина_волны": transformed_x, "Интенсивность": transformed_y})

        # Сохранение в базу данных
        self.spectrum_data_frame.load_extracted_data(df, self.x_min, self.x_max, self.y_min, self.y_max, is_descending=True, is_ai_digitized=True)
        logger.info("График успешно оцифрован и сохранён")
        self.close()

    def average_y_values(self, x, y):
        if len(x) == 0:
            return [], []
        x_unique = sorted(list(set(x)))
        y_averaged = [np.mean([y[i] for i, val in enumerate(x) if val == x_val]) for x_val in x_unique]
        return x_unique, y_averaged