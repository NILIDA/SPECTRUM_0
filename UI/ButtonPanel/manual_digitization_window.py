# UI/ButtonPanel/manual_digitization_window.py
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QInputDialog, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

class ManualDigitizationWindow(QMainWindow):
    def __init__(self, spectrum_data_frame):
        super().__init__()
        self.spectrum_data_frame = spectrum_data_frame
        self.setWindowTitle("Ручная оцифровка графика")
        self.setGeometry(100, 100, 800, 600)

        # Создаем Matplotlib canvas для отображения изображения
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.image = None
        self.points = []
        self.reference_points = {'x': None, 'y': None}
        self.spline_line = None  # Для хранения линии сплайна

        # Настройка интерфейса
        self.layout = QVBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.load_button = QPushButton("Загрузить изображение")
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        self.set_reference_button = QPushButton("Установить опорные точки")
        self.set_reference_button.clicked.connect(self.set_reference_points)
        self.layout.addWidget(self.set_reference_button)

        self.digitize_button = QPushButton("Оцифровать график")
        self.digitize_button.clicked.connect(self.digitize_curve)
        self.layout.addWidget(self.digitize_button)

        self.save_button = QPushButton("Сохранить данные")
        self.save_button.clicked.connect(self.save_data)
        self.layout.addWidget(self.save_button)

        self.layout.addWidget(self.canvas)

        # Подключение обработчиков событий мыши
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def load_image(self):
        """Загрузка изображения."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.image = plt.imread(file_name)
            self.ax.imshow(self.image)
            self.canvas.draw()

    def set_reference_points(self):
        """Установка опорных точек для осей X и Y."""
        if self.image is None:
            return

        # Установка опорных точек для оси X
        self.ax.set_title("Выберите НАЧАЛО оси X")
        self.canvas.draw()
        coord1 = self.get_point()
        data1, ok = QInputDialog.getDouble(self, "Введите МАКСИМУМ", "Введите максимальное значение для оси X (например, 4000)")
        if not ok:
            return

        self.ax.set_title("Выберите КОНЕЦ оси X")
        self.canvas.draw()
        coord2 = self.get_point()
        data2, ok = QInputDialog.getDouble(self, "Введите МИНИМУМ", "Введите минимальное значение для оси X (например, 0)")
        if not ok:
            return

        self.reference_points['x'] = {
            'coord1': coord1, 'data1': data1,
            'coord2': coord2, 'data2': data2
        }

        # Установка опорных точек для оси Y
        self.ax.set_title("Выберите НАЧАЛО оси Y")
        self.canvas.draw()
        coord1 = self.get_point()
        data1, ok = QInputDialog.getDouble(self, "Введите МИНИМУМ", "Введите минимальное значение для оси Y")
        if not ok:
            return

        self.ax.set_title("Выберите КОНЕЦ оси Y")
        self.canvas.draw()
        coord2 = self.get_point()
        data2, ok = QInputDialog.getDouble(self, "Введите МАКСИМУМ", "Введите максимальное значение для оси Y")
        if not ok:
            return

        self.reference_points['y'] = {
            'coord1': coord1, 'data1': data1,
            'coord2': coord2, 'data2': data2
        }

        self.ax.set_title("Опорные точки установлены")
        self.canvas.draw()

    def get_point(self):
        """Получение координат точки от пользователя."""
        point = None
        def on_click(event):
            nonlocal point
            if event.inaxes == self.ax:
                point = (event.xdata, event.ydata)
                self.canvas.mpl_disconnect(cid)
        cid = self.canvas.mpl_connect('button_press_event', on_click)
        while point is None:
            plt.pause(0.1)
        return point

    def digitize_curve(self):
        """Оцифровка графика пользователем."""
        if self.image is None or self.reference_points['x'] is None or self.reference_points['y'] is None:
            return

        self.ax.set_title("Оцифруйте график: ЛКМ — добавить точку, ПКМ — удалить, СКМ — завершить")
        self.canvas.draw()

        self.points = []
        self.spline_line = None  # Для хранения линии сплайна

        def on_click(event):
            if event.inaxes == self.ax:
                if event.button == 1:  # Левый клик — добавить точку
                    self.points.append((event.xdata, event.ydata))
                    self.ax.plot(event.xdata, event.ydata, 'go')  # Отрисовка точки
                    self.update_spline()  # Обновление сплайна
                elif event.button == 3:  # Правый клик — удалить последнюю точку
                    if self.points:
                        self.points.pop()
                        # Удаляем все линии (точки и сплайн)
                        for line in self.ax.lines:
                            line.remove()
                        # Перерисовываем все точки
                        for point in self.points:
                            self.ax.plot(point[0], point[1], 'go')
                        self.update_spline()  # Обновление сплайна
                elif event.button == 2:  # Средний клик — завершить
                    self.canvas.mpl_disconnect(cid)
                    # Очищаем сплайн перед завершением
                    if self.spline_line:
                        self.spline_line.remove()
                        self.spline_line = None
                    self.canvas.draw()
                self.canvas.draw()

        cid = self.canvas.mpl_connect('button_press_event', on_click)
        plt.show(block=True)

    def update_spline(self):
        """Обновление сплайновой интерполяции между точками."""
        if len(self.points) < 2:  # Сплайн требует минимум 2 точки
            if self.spline_line:  # Удаляем сплайн, если он был
                self.spline_line.remove()
                self.spline_line = None
            self.canvas.draw()
            return

        # Извлекаем x и y координаты точек
        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])

        # Сортировка по x для корректной интерполяции
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # Создание сплайна
        try:
            spline = UnivariateSpline(x_sorted, y_sorted, s=0, k=3)  # s=0 для точной интерполяции, k=3 для кубического сплайна
            x_smooth = np.linspace(min(x_sorted), max(x_sorted), 100)  # Гладкая ось x
            y_smooth = spline(x_smooth)

            # Удаляем предыдущий сплайн, если он существует
            if self.spline_line:
                self.spline_line.remove()

            # Отрисовка нового сплайна
            self.spline_line, = self.ax.plot(x_smooth, y_smooth, 'b-', linewidth=1)  # Синяя линия сплайна
            self.canvas.draw()
        except Exception as e:
            print(f"Ошибка при построении сплайна: {e}")

    def save_data(self):
        """Сохранение оцифрованных данных в базу данных."""
        if not self.points:
            return

        # Преобразование координат из пикселей в данные
        x_ref = self.reference_points['x']
        y_ref = self.reference_points['y']

        x_pixel_range = x_ref['coord2'][0] - x_ref['coord1'][0]
        x_data_range = x_ref['data2'] - x_ref['data1']  # data2 - минимальное, data1 - максимальное
        x_scale = x_data_range / x_pixel_range

        y_pixel_range = y_ref['coord2'][1] - y_ref['coord1'][1]
        y_data_range = y_ref['data2'] - y_ref['data1']
        y_scale = y_data_range / y_pixel_range

        data = []
        for point in self.points:
            # Учитываем, что ось X убывающая (от максимума к минимуму)
            x_data = x_ref['data1'] + (point[0] - x_ref['coord1'][0]) * x_scale
            y_data = y_ref['data1'] + (point[1] - y_ref['coord1'][1]) * y_scale
            data.append([x_data, y_data])

        # Создание DataFrame
        df = pd.DataFrame(data, columns=["Длина_волны", "Интенсивность"])

        # Сортировка по убыванию длины волны
        df = df.sort_values(by="Длина_волны", ascending=False)

        # Определение направления оси X
        is_descending = True  # Всегда True для ИК-спектров

        # Сохранение в базу данных через существующий метод
        self.spectrum_data_frame.load_extracted_data(
            df, x_ref['data1'], x_ref['data2'], y_ref['data1'], y_ref['data2'], is_descending
        )
        self.close()

    def on_click(self, event):
        """Пустой обработчик событий мыши (обрабатывается в других методах)."""
        pass