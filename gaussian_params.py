import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal
from logger_config import logger

from PyQt6.QtCore import QObject, pyqtSignal


class GaussianParams(QObject):
    data_changed_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()        
        self.gaussian_df = pd.DataFrame(columns=['Height', 'Position', 'Width'])
            
    def update_params(self, df):
        logger.debug(f"Вызван слот update_params с данными: {df}")
        if 'Height' in df.columns:
            self.gaussian_df = df            
            self.data_changed_signal.emit()
            logger.debug(f"Данные значения гауссиан изменены изменены")            
            
    def reset_df(self):
        self.gaussian_df = pd.DataFrame(columns=['Height', 'Position', 'Width'])
        
    def concat_new_gaussian(self, df):
        self.gaussian_df = pd.concat([self.gaussian_df, df], ignore_index=True)