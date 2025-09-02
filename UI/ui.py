# UI/ui.py
from PyQt6.QtWidgets import QWidget, QHBoxLayout
from PyQt6.QtCore import pyqtSignal
from .GraphicalArea.graphical_area import GraphicalArea
from .ButtonPanel.button_panel import ButtonPanel
from logger_config import logger

class UI(QWidget):
    graphical_area_shown_signal = pyqtSignal()  # Сигнал для отображения страницы графика

    def __init__(self, spectrum_data_frame, config, gaussian_params):
        super().__init__()
        self.spectrum_data_frame = spectrum_data_frame
        self.config = config
        self.gaussian_params = gaussian_params
        self.graphical_area = GraphicalArea(self.config, self.gaussian_params, self)
        self.button_panel = ButtonPanel(self.spectrum_data_frame, self.config, self.gaussian_params)
        layout = QHBoxLayout()
        layout.addWidget(self.graphical_area)
        layout.addWidget(self.button_panel)
        self.setLayout(layout)
        # Вызов метода для эмуляции отображения графика при инициализации
        self.show_graphical_area()

    def show_graphical_area(self):
        """Метод для отображения страницы с графиком."""
        self.graphical_area.setVisible(True)
        self.graphical_area_shown_signal.emit()
