# UI/ButtonPanel/load_button.py
from PyQt6.QtWidgets import QPushButton
from .manual_digitization_window import ManualDigitizationWindow
from .ai_digitization_window import AIDigitizationWindow
from .load_image.graph_extraction_window import GraphExtractionWindow  # Добавляем импорт

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
        self.spectrum_data_frame = spectrum_data_frame
        self.clicked.connect(self.on_click)

    def on_click(self):
        self.graphExtractor = GraphExtractionWindow(self.spectrum_data_frame)
        self.graphExtractor.show()

class ManualDigitizeButton(QPushButton):
    def __init__(self, spectrum_data_frame):
        super().__init__('Оцифровать график вручную')
        self.spectrum_data_frame = spectrum_data_frame
        self.clicked.connect(self.on_click)

    def on_click(self):
        self.manual_digitization_window = ManualDigitizationWindow(self.spectrum_data_frame)
        self.manual_digitization_window.show()

class AIDigitizeButton(QPushButton):
    def __init__(self, spectrum_data_frame):
        super().__init__('Оцифровать с помощью ИИ')
        self.spectrum_data_frame = spectrum_data_frame
        self.clicked.connect(self.on_click)

    def on_click(self):
        self.ai_digitization_window = AIDigitizationWindow(self.spectrum_data_frame)
        self.ai_digitization_window.show()