from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject
import pandas as pd
from logger_config import logger
import cv2
from read_image import img2df, graph2df


class SpectrumDataFrame(QObject):
    def __init__(self):
        super().__init__()
        self.column_names = ["Длина_волны", "Интенсивность"]
        
    spectrum_loaded_signal = pyqtSignal(pd.DataFrame)
    plot_spectrum_signal = pyqtSignal(pd.DataFrame, list)
    
    @pyqtSlot()
    def plot_spectrum(self):
        self.plot_spectrum_signal.emit(self.df, self.column_names)
        
    def load_spectrum_txt(self):        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(caption="Open Text File", filter="Text Files (*.txt)")
        if file_path:
            try:
                self.df = pd.read_csv(
                    file_path, delim_whitespace=True, header=None, skiprows=1, names=self.column_names)
                self.spectrum_loaded_signal.emit(self.df)
                self.plot_spectrum()
                logger.debug(f"data head {self.df.head()}")
                return self.df
            except Exception as e:
                logger.warning(f"Ошибка при загрузке файла: {e}")
                return None
        else:
            logger.info("Файл не выбран")
            return None

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName()
        if file_path:
            try:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                clr = cv2.imread(file_path)
                if image is not None:
                    self.df = img2df(image, clr, 175, 5, 100, 57, 170, 220, 180, 220, 5)
                    self.spectrum_loaded_signal.emit(self.df)
                    self.plot_spectrum()
                else:
                    logger.warning(f"Ошибка при загрузке изображения")
            except Exception as e:
                logger.warning(f"Ошибка при загрузке изображения: {e}")
                return None
        else:
            logger.info("Файл не выбран")
            return None

    def load_edit_image(self, img):
        if img is not None:
            self.df = graph2df(img)
            self.spectrum_loaded_signal.emit(self.df)
            self.plot_spectrum()
        else:
            logger.warning(f"Ошибка при загрузке изображения")



    @pyqtSlot(tuple)
    def subctract_slice_background(self, slice_borders: tuple):
        logger.debug(f"Получены новые точки диапазона интегрирования: {slice_borders}")
        lower_bound, upper_bound = sorted(slice_borders)        
        # Фильтрация DataFrame
        filtered_df = self.df[(self.df["Длина_волны"] >= lower_bound) & (self.df["Длина_волны"] <= upper_bound)].copy()        
        # Вычисление уравнения прямой
        start_point = filtered_df.iloc[0]
        end_point = filtered_df.iloc[-1]
        m = ((end_point["Интенсивность"] - start_point["Интенсивность"]) 
             / (end_point["Длина_волны"] - start_point["Длина_волны"]))
        c = start_point["Интенсивность"] - m * start_point["Длина_волны"]        
        # Нахождение максимального значения для горизонтальной линии
        max_value = filtered_df["Интенсивность"].max()        
        # Выравнивание и вычитание фона
        filtered_df["Интенсивность"] = filtered_df.apply(
            lambda row: row["Интенсивность"] - (m * row["Длина_волны"] + c) + max_value, axis=1)        
        filtered_df["Интенсивность"] = filtered_df["Интенсивность"] + (100 - filtered_df["Интенсивность"].max())
        filtered_df["Интенсивность"] = filtered_df["Интенсивность"] - filtered_df["Интенсивность"].max()        
        self.plot_spectrum_signal.emit(filtered_df, self.column_names)
        
       