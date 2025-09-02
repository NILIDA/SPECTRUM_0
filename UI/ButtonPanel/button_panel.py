# UI/ButtonPanel/button_panel.py
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QPushButton
from .load_button import LoadButton, LoadImg, LoadExtImg, ManualDigitizeButton, AIDigitizeButton  # Добавляем AIDigitizeButton
from .spectrum_table import SpectrumTable
from db_window import DBWindow

class ButtonPanel(QWidget):
    def __init__(self, spectrum_data_frame, config, gaussian_params):
        super().__init__()
        self.spectrum_data_frame = spectrum_data_frame
        button_layout = QVBoxLayout()
        self.load_button = LoadButton(spectrum_data_frame)
        self.load_image = LoadExtImg(spectrum_data_frame)
        self.manual_digitize_button = ManualDigitizeButton(spectrum_data_frame)
        self.ai_digitize_button = AIDigitizeButton(spectrum_data_frame)
        self.db_button = QPushButton("Открыть базу данных")
        self.db_button.clicked.connect(self.open_db_window)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.load_image)
        button_layout.addWidget(self.manual_digitize_button)
        button_layout.addWidget(self.ai_digitize_button)
        button_layout.addWidget(self.db_button)
        table_layout = QVBoxLayout()
        self.spectrum_table = SpectrumTable(config, gaussian_params)
        table_layout.addWidget(self.spectrum_table)
        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addStretch(1)
        main_layout.addLayout(table_layout)
        self.setLayout(main_layout)

    def open_db_window(self):
        self.db_window = DBWindow(self.spectrum_data_frame)
        self.db_window.show()