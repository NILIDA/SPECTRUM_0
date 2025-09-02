# UI/GraphicalArea/graphical_area.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSlot, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from config import SpectrumConfig
from .custom_toolbar import CustomToolbar
from .integral_action_callbacks import IntegralActionCallbacks
from .gauss_action_callbacks import GaussActionCallbacks
from logger_config import logger
plt.style.use('default')

class GraphicalArea(QWidget):
    gauss_released_signal = pyqtSignal(pd.DataFrame)
    mouse_released_signal = pyqtSignal(tuple)

    def __init__(self, config, gaussian_params, graphical_area=None):
        super().__init__(graphical_area)
        self.config = config
        self.gaussian_params = gaussian_params
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = CustomToolbar(self.canvas, self)
        self.integral_callbacks = IntegralActionCallbacks(self)
        self.gauss_callbacks = GaussActionCallbacks(self)
        self.shading_regions = []
        self.original_xlim = None
        self.press_x = None
        self.press_y = None
        self.mouse_pressed = False
        self.x_data = None
        self.y_data = None
        self.is_descending = True
        self.is_ai_digitized = False  # Флаг для ИИ-оцифровки
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear_plot(self):
        """Очищает график и сбрасывает все данные."""
        self.ax.clear()
        self.shading_regions = []
        self.x_data = None
        self.y_data = None
        self.original_xlim = None
        self.canvas.draw()
        logger.debug("График очищен")

    def on_mouse_press(self, event):
        if event.inaxes:
            if event.button == 3 or event.button == 2:
                self.toolbar.deactivate_all_actions()
            if self.toolbar._actions['pan'].isChecked():
                return  # Блокируем интегрирование и гауссы, если "Pan" активна
            if self.toolbar.integral_action.isChecked():
                if event.button == 1:
                    self.integral_callbacks.on_press(event)
            if self.toolbar.gauss_action.isChecked():
                if event.button == 1:
                    self.gauss_callbacks.on_press(event)

    def on_mouse_move(self, event):
        if self.toolbar._actions['pan'].isChecked():
            return  # Блокируем интегрирование и гауссы, если "Pan" активна
        if self.toolbar.integral_action.isChecked() and self.mouse_pressed and event.xdata and event.inaxes:
            self.integral_callbacks.on_move(event)
        if self.toolbar.gauss_action.isChecked() and self.mouse_pressed and event.xdata and event.inaxes:
            self.gauss_callbacks.on_move(event)

    def on_mouse_release(self, event):
        if self.toolbar._actions['pan'].isChecked():
            return  # Блокируем интегрирование и гауссы, если "Pan" активна
        if self.toolbar.integral_action.isChecked() and event.inaxes:
            self.integral_callbacks.on_release(event)
        if self.toolbar.gauss_action.isChecked() and event.inaxes:
            self.gauss_callbacks.on_release(event)
            self.gauss_released_signal.emit(self.gaussian_params.gaussian_df)

    def draw_curves(self):
        logger.debug("Сигнал об отрисовке кривых получен")
        self.gauss_callbacks.draw_gaussian_curves()

    @pyqtSlot(pd.DataFrame, list, bool, float, float, float, float, bool)
    def plot_data(self, df, column_names, is_descending, x_min_val, x_max_val, y_min_val, y_max_val, is_ai_digitized):
        logger.debug(f"Получен сигнал plot_data с DataFrame:\n{df.head()}")
        logger.debug(f"column_names: {column_names}, is_descending: {is_descending}, x_min_val: {x_min_val}, x_max_val: {x_max_val}, y_min_val: {y_min_val}, y_max_val: {y_max_val}, is_ai_digitized: {is_ai_digitized}")
        if not all(col in df.columns for col in column_names):
            logger.error(f"Отсутствуют ожидаемые столбцы {column_names} в DataFrame. Имеются: {df.columns.tolist()}")
            return
        if df.empty:
            logger.error("DataFrame пустой, график не может быть построен")
            return
        self.is_descending = is_descending
        self.is_ai_digitized = is_ai_digitized  # Сохраняем флаг
        self.ax.clear()
        self.shading_regions = []
        x_data = df[column_names[0]]
        y_data = df[column_names[1]]
        try:
            window_length = max(3, self.config.Savitzky_df['window_length'].astype(int).item())
            polyorder = min(self.config.Savitzky_df['polyorder'].astype(int).item(), window_length - 1)
            y_data_smooth = signal.savgol_filter(
                y_data.to_numpy(),
                window_length=window_length,
                polyorder=polyorder,
                mode=self.config.Savitzky_df['Savitzky_mode'].astype(str).item()
            )
        except Exception as e:
            logger.error(f"Ошибка в savgol_filter: {e}")
            y_data_smooth = y_data.to_numpy()
        sorted_indices = x_data.argsort()[::-1] if is_descending else x_data.argsort()
        self.x_data = x_data.iloc[sorted_indices]
        self.y_data = y_data_smooth[sorted_indices]
        # Применяем смещение, если данные оцифрованы с помощью ИИ

        self.ax.plot(self.x_data, self.y_data)
        self.ax.set_xlim(max(x_max_val, x_min_val), min(x_max_val, x_min_val))
        self.ax.set_ylim(y_min_val, y_max_val)
        self.ax.set_xlabel(column_names[0])
        self.ax.set_ylabel(column_names[1])
        self.ax.grid(True)
        logger.debug(f"Значения Savitzky_df в plot_data: {self.config.Savitzky_df}")
        logger.debug(f"x_data (первые 5): {self.x_data[:5].tolist()}")
        logger.debug(f"y_data (первые 5): {self.y_data[:5].tolist()}")
        self.canvas.draw()
        logger.debug("Холст перерисован")