# main.py
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from UI.ui import UI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from spectrum_data_frame import SpectrumDataFrame
from config import SpectrumConfig
from gaussian_params import GaussianParams
from logger_config import logger
from UI.ButtonPanel.load_image.edit_img import ColorEditWindow
from read_image import graph2df

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.spectrum_data_frame = SpectrumDataFrame()
        self.config = SpectrumConfig()
        self.gaussian_params = GaussianParams()
        self.ui = UI(self.spectrum_data_frame, self.config, self.gaussian_params)
        self.setCentralWidget(self.ui)
        self.setWindowTitle('Spectrum Analysis Tool')
        # Подключение сигналов и слотов
        self.spectrum_data_frame.spectrum_loaded_signal.connect(
            self.ui.button_panel.spectrum_table.update_table)
        self.ui.graphical_area.toolbar.integral_action_signal.connect(
            self.ui.button_panel.spectrum_table.update_table)
        self.ui.graphical_area.toolbar.gauss_action_signal.connect(
            self.ui.button_panel.spectrum_table.update_table)
        self.ui.graphical_area.gauss_released_signal.connect(
            self.ui.button_panel.spectrum_table.update_table)
        self.spectrum_data_frame.plot_spectrum_signal.connect(
            self.ui.graphical_area.plot_data)
        self.ui.graphical_area.mouse_released_signal.connect(
            self.spectrum_data_frame.subctract_slice_background)
        # Подключение сигнала кнопки "Домой" для полного сброса
        self.ui.graphical_area.toolbar.home_signal.connect(
            self.restore_original_state)
        # Подключение сигнала кнопки "Окно графика" для восстановления
        self.ui.graphical_area.toolbar.restore_graphical_view_signal.connect(
            self.restore_graphical_view)
        self.gaussian_params.data_changed_signal.connect(
            self.ui.graphical_area.draw_curves)
        self.color_edit_window = ColorEditWindow(None)

    def restore_original_state(self):
        """Сбрасывает программу в начальное состояние при нажатии 'Домой'."""
        self.spectrum_data_frame.reset_data()  # Очистка данных
        self.ui.graphical_area.clear_plot()  # Очистка графика
        self.ui.button_panel.spectrum_table.clear_table()  # Очистка таблицы
        self.ui.graphical_area.gauss_callbacks.reset_gaussian_params()  # Сброс параметров Гаусса
        logger.debug("Программа сброшена в начальное состояние: график и таблица очищены")

    def restore_graphical_view(self):
        """Восстанавливает график и таблицу при нажатии 'Окно графика'."""
        if self.spectrum_data_frame.df is not None and not self.spectrum_data_frame.df.empty:
            self.spectrum_data_frame.plot_spectrum()  # Перерисовка текущего графика
            self.ui.button_panel.spectrum_table.update_table(self.spectrum_data_frame.df)  # Обновление таблицы
            logger.debug("График и таблица восстановлены")

    def closeEvent(self, event):
        self.spectrum_data_frame.close_db()
        event.accept()

def main():
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()