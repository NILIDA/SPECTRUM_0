import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from UI.ui import UI  # Импорт пользовательского интерфейса
import matplotlib.pyplot as plt  # Импорт библиотеки для построения графиков
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Импорт интеграции графиков Matplotlib с PyQt
from spectrum_data_frame import SpectrumDataFrame  # Импорт класса для работы с данными спектра
from config import SpectrumConfig  # Импорт класса для работы с конфигурацией
from gaussian_params import GaussianParams  # Импорт класса для работы с параметрами гауссиана
from logger_config import logger  # Импорт настройки логгера
from UI.ButtonPanel.load_image.edit_img import ColorEditWindow
from read_image import graph2df

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.spectrum_data_frame = SpectrumDataFrame()  # Создание объекта для работы с данными спектра
        self.config = SpectrumConfig()  # Создание объекта для работы с конфигурацией
        self.gaussian_params = GaussianParams()  # Создание объекта для работы с параметрами гауссиана
        self.ui = UI(self.spectrum_data_frame, self.config, self.gaussian_params)  # Создание пользовательского интерфейса
        self.setCentralWidget(self.ui)  # Установка пользовательского интерфейса в центр окна
        self.setWindowTitle('Spectrum Analysis Tool')  # Установка заголовка окна
        # Подключение сигналов и слотов для обновления данных в таблице спектра
        self.spectrum_data_frame.spectrum_loaded_signal.connect(
            self.ui.button_panel.spectrum_table.update_table)
        self.ui.graphical_area.toolbar.integral_action_signal.connect(
            self.ui.button_panel.spectrum_table.update_table)
        self.ui.graphical_area.toolbar.gauss_action_signal.connect(
            self.ui.button_panel.spectrum_table.update_table)
        self.ui.graphical_area.gauss_released_signal.connect(
            self.ui.button_panel.spectrum_table.update_table)
        # Подключение сигналов и слотов для построения графиков спектра
        self.spectrum_data_frame.plot_spectrum_signal.connect(
            self.ui.graphical_area.plot_data)
        self.ui.graphical_area.mouse_released_signal.connect(
            self.spectrum_data_frame.subctract_slice_background)
        self.ui.graphical_area.toolbar.restore_df_plot_signal.connect(
            self.spectrum_data_frame.plot_spectrum)
        self.ui.graphical_area.toolbar.restore_df_plot_signal.connect(
            self.ui.graphical_area.gauss_callbacks.reset_gaussian_params)
        # Подключение сигналов и слотов для обновления графического интерфейса при изменении данных гауссиана
        self.gaussian_params.data_changed_signal.connect(
            self.ui.graphical_area.draw_curves
        )

        #сигналы для расширенной обработки изображения
        # Создаем экземпляр ColorEditWindow
        self.color_edit_window = ColorEditWindow(None)

def main():
    app = QApplication(sys.argv)  # Создание объекта приложения Qt
    main = Main()  # Создание главного окна
    main.show()  # Отображение главного окна
    sys.exit(app.exec())  # Запуск цикла обработки событий приложения

if __name__ == '__main__':
    main()
