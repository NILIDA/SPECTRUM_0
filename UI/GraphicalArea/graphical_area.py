from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSlot, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from scipy import signal
from config import SpectrumConfig
from .custom_toolbar import CustomToolbar
from .integral_action_callbacks import IntegralActionCallbacks
from .gauss_action_callbacks import GaussActionCallbacks
from logger_config import logger
plt.style.use(['science', 'no-latex', 'nature', 'grid'])


class GraphicalArea(QWidget):
    
    # Определение сигналов
    gauss_released_signal = pyqtSignal(pd.DataFrame)
    mouse_released_signal = pyqtSignal(tuple)
    
    def __init__(self, config, gaussian_params, graphical_area=None):
        super().__init__(graphical_area)
        self.config = config
        self.gaussian_params = gaussian_params
        # Создание фигуры и осей для графика
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)        
        # Создание пользовательской панели инструментов
        self.toolbar = CustomToolbar(self.canvas, self)
        # настройка поведения кастомных действий панели инструментов
        self.integral_callbacks = IntegralActionCallbacks(self)
        self.gauss_callbacks = GaussActionCallbacks(self)
        # Инициализация переменных для управления затенением и масштабом
        self.shading_regions = []  # Список для хранения затемненных областей
        self.original_xlim = None  # Исходный масштаб оси X
        self.press_x = None        # Координата X при нажатии мыши
        self.press_y = None        # Координата Y при нажатии мыши
        self.mouse_pressed = False # Флаг состояния нажатия мыши
        # Подключение обработчиков событий мыши
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move) 
        # Размещение элементов в макете
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def on_mouse_press(self, event):
        if event.inaxes:
            if event.button == 3 or event.button == 2:
                self.toolbar.deactivate_all_actions()
            if self.toolbar.integral_action.isChecked():
                if event.button == 1:
                    self.integral_callbacks.on_press(event)
            if self.toolbar.gauss_action.isChecked():
                if event.button == 1:
                    self.gauss_callbacks.on_press(event)                            
    
    def on_mouse_move(self, event):
        if self.toolbar.integral_action.isChecked() and self.mouse_pressed and event.xdata and event.inaxes:
            self.integral_callbacks.on_move(event)
        if self.toolbar.gauss_action.isChecked() and self.mouse_pressed and event.xdata and event.inaxes:
            self.gauss_callbacks.on_move(event)
    
    def on_mouse_release(self, event):
        if self.toolbar.integral_action.isChecked() and event.inaxes:
            self.integral_callbacks.on_release(event)                       
        if self.toolbar.gauss_action.isChecked() and event.inaxes:
            self.gauss_callbacks.on_release(event) 
            self.gauss_released_signal.emit(self.gaussian_params.gaussian_df)
            
    def draw_curves(self):
        logger.debug("Сигнал об отрисовке кривых получен")
        self.gauss_callbacks.draw_gaussian_curves()
        
    # Слот для отображения данных на графике
    @pyqtSlot(pd.DataFrame, list)
    def plot_data(self, df, column_names):        
        self.ax.clear()
        self.x_data = df[column_names[0]]
        self.y_data = signal.savgol_filter(
            df[column_names[1]], 
            window_length=self.config.Savitzky_df['window_length'].astype(int).item(),
            polyorder=self.config.Savitzky_df['polyorder'].astype(int).item(), 
            mode=self.config.Savitzky_df['Savitzky_mode'].astype(str).item()) 
        logger.debug(f"Значения Savitzky_df в plot_data: {self.config.Savitzky_df}")       
        # Построение графика по заданным данным
        self.ax.plot(self.x_data, self.y_data)          
        self.canvas.draw()

