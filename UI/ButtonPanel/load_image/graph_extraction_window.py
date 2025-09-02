# UI/ButtonPanel/load_image/graph_extraction_window.py
import cv2
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtCore import Qt
from scipy.signal import savgol_filter
import pandas as pd
from logger_config import logger

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
            y_min = float(self.y_min_input.text()) if self.y_min_input.text() else 0
            y_max = float(self.y_max_input.text()) if self.y_max_input.text() else 100
            return x_min, x_max, y_min, y_max
        except ValueError:
            logger.error("Ошибка: Введите числовые значения для осей")
            return 0, 4000, 0, 100

class GraphExtractionWindow(QMainWindow):
    def __init__(self, spectrum_data_frame):
        super().__init__()
        self.spectrum_data_frame = spectrum_data_frame
        self.setWindowTitle("Извлечение графика")
        self.image = None
        self.run_extraction()

    def detect_graph_bounds(self, image):
        """Автоматическое определение границ графика на изображении."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            return 0, 0, image.shape[1], image.shape[0]
        
        x, y, w, h = graph_rect
        return x, y, x + w, y + h

    def run_extraction(self):
        # Шаг 1: Загрузка изображения
        file_name, _ = QFileDialog.getOpenFileName(self, "Выбери изображение", "", "Image Files (*.png *.jpg *.jpeg)")
        if not file_name:
            logger.info("Изображение не выбрано")
            self.close()
            return
        self.image = cv2.imread(file_name)
        if self.image is None:
            logger.error(f"Ошибка: не удалось загрузить изображение {file_name}")
            self.close()
            return

        # Шаг 2: Открытие диалогового окна для ввода значений осей
        dialog = AxisInputDialog(self)
        if dialog.exec():
            x_min_val, x_max_val, y_min_val, y_max_val = dialog.get_inputs()
        else:
            logger.info("Ввод значений осей отменён")
            self.close()
            return

        # Шаг 3: Выбор цвета графика
        logger.info("Выберите цвет линии графика:")
        graph_color = self.pick_color_rgb()

        # Шаг 4: Рисование маски
        logger.info("Нарисуйте маску вокруг линии графика (q — завершить, c — очистить):")
        drawn_mask = self.draw_mask()

        # Шаг 5: Обработка изображения с маской
        mask_img = self.preprocess_image_with_mask(graph_color, drawn_mask)
        contours = self.find_graph_contours(mask_img)
        if not contours:
            logger.error("Контуры графика не найдены.")
            self.close()
            return

        # Шаг 6: Извлечение точек графика
        all_x, all_y = [], []
        for contour in contours:
            x, y = self.extract_graph_points(contour)
            all_x.extend(x)
            all_y.extend(y)
        all_x, all_y = self.average_y_values(all_x, all_y)
        all_x, all_y = np.array(all_x), np.array(all_y)

        # Шаг 7: Автоматическое определение границ графика
        min_x_pix, min_y_pix, max_x_pix, max_y_pix = self.detect_graph_bounds(self.image)

        # Для ИК-спектров ось X всегда убывающая
        is_descending = True

        # Проверка на наличие точек
        if len(all_x) == 0:
            logger.error("Не найдено точек графика после обработки.")
            self.close()
            return

        # Преобразование пиксельных координат в значения длины волны
        x_range = x_max_val - x_min_val
        y_range = y_max_val - y_min_val
        transformed_x = x_max_val - (all_x - min_x_pix) / (max_x_pix - min_x_pix) * x_range
        transformed_y = y_min_val + (all_y - min_y_pix) / (max_y_pix - min_y_pix) * y_range

        # Ограничение значений
        transformed_x = np.clip(transformed_x, x_min_val, x_max_val)
        transformed_y = np.clip(transformed_y, y_min_val, y_max_val)

        # Применение фильтра Савицкого-Голея для сглаживания (если достаточно точек)
        if len(transformed_y) >= 11:
            transformed_y = savgol_filter(transformed_y, window_length=11, polyorder=2)

        # Сортировка по убыванию X (для ИК-спектров)
        sorted_indices = np.argsort(transformed_x)[::-1]
        transformed_x = transformed_x[sorted_indices]
        transformed_y = transformed_y[sorted_indices]

        # Создание DataFrame и загрузка
        df = pd.DataFrame({"Длина_волны": transformed_x, "Интенсивность": transformed_y})
        self.spectrum_data_frame.load_extracted_data(df, x_min_val, x_max_val, y_min_val, y_max_val, is_descending)
        logger.info("График успешно извлечён и сохранён")
        self.close()

    def pick_color_rgb(self):
        selected_color = [0, 0, 0]
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_color[:] = self.image[y, x][::-1]
                logger.info(f"Выбранный цвет RGB: {selected_color}")
        cv2.imshow("Выберите цвет", self.image)
        cv2.setMouseCallback("Выберите цвет", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return selected_color

    def draw_mask(self):
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        drawing = False
        erasing = False
        thickness = 15
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, erasing
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                cv2.circle(mask, (x, y), thickness, 255, -1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                erasing = True
                cv2.circle(mask, (x, y), thickness, 0, -1)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.circle(mask, (x, y), thickness, 255, -1)
                elif erasing:
                    cv2.circle(mask, (x, y), thickness, 0, -1)
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                drawing = erasing = False

        cv2.namedWindow("Нарисуйте маску", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Нарисуйте маску", 400, 300)
        cv2.setMouseCallback("Нарисуйте маску", mouse_callback)
        logger.info("ЛКМ — рисовать, ПКМ — стирать, 'q' — завершить, 'c' — очистить")
        while True:
            combined = cv2.addWeighted(self.image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
            cv2.imshow("Нарисуйте маску", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                mask[:] = 0
        cv2.destroyAllWindows()
        return mask

    def preprocess_image_with_mask(self, graph_color, drawn_mask):
        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        graph_color_hsv = cv2.cvtColor(np.uint8([[graph_color]]), cv2.COLOR_RGB2HSV)[0][0]
        lower_hsv = np.clip(graph_color_hsv - [30, 80, 80], [0, 0, 0], [180, 255, 255])
        upper_hsv = np.clip(graph_color_hsv + [30, 80, 80], [0, 0, 0], [180, 255, 255])
        graph_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        final_mask = cv2.bitwise_and(graph_mask, graph_mask, mask=drawn_mask)
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return final_mask

    def find_graph_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.contourArea(cnt) > 5]

    def extract_graph_points(self, contour):
        x, y = [], []
        for point in contour:
            x.append(point[0][0])
            y.append(self.image.shape[0] - point[0][1])
        return x, y

    def average_y_values(self, x, y):
        if len(x) == 0:
            return [], []
        x_unique = sorted(list(set(x)))
        y_averaged = [np.mean([y[i] for i, val in enumerate(x) if val == x_val]) for x_val in x_unique]
        return x_unique, y_averaged