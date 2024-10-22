import pandas as pd
from PyQt6.QtCore import pyqtSlot
from logger_config import logger

class SpectrumConfig:
    def __init__(self):
        self.Savitzky_df = pd.DataFrame({'window_length':[1],'polyorder':[0], 'Savitzky_mode':['nearest']})
            
    def update_config(self, df):
        logger.debug(f"Вызван слот update_config с данными: {df}")
        if 'window_length' in df.columns:
            self.Savitzky_df = df
            logger.debug(f"Данные настроек сглаживания Савицкого изменены")