# spectrum_data_frame.py
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject
import pandas as pd
import numpy as np
from logger_config import logger
import cv2
from read_image import img2df, graph2df
from db_manager import DBManager
import os

class SpectrumDataFrame(QObject):
    spectrum_loaded_signal = pyqtSignal(pd.DataFrame)
    plot_spectrum_signal = pyqtSignal(pd.DataFrame, list, bool, float, float, float, float, bool)  # Добавляем bool для is_ai_digitized

    def __init__(self):
        super().__init__()
        self.column_names = ["Длина_волны", "Интенсивность"]
        self.db = DBManager()
        self.df = None
        self.is_ai_digitized = False  # Флаг для ИИ-оцифровки

    def reset_data(self):
        """Очищает DataFrame, подготавливая программу к новому сканированию."""
        self.df = None
        self.is_ai_digitized = False
        logger.debug("Данные в SpectrumDataFrame сброшены (self.df = None)")

    @pyqtSlot()
    def plot_spectrum(self):
        if self.df is None or self.df.empty:
            logger.error("DataFrame пустой или не инициализирован. Нельзя отобразить график.")
            return
        x_min_val = self.df['Длина_волны'].min()
        x_max_val = self.df['Длина_волны'].max()
        y_min_val = self.df['Интенсивность'].min()
        y_max_val = self.df['Интенсивность'].max()
        logger.debug(f"Отправка сигнала plot_spectrum_signal с DataFrame:\n{self.df.head()}")
        logger.debug(f"Пределы осей: x_min={x_min_val}, x_max={x_max_val}, y_min={y_min_val}, y_max={y_max_val}")
        self.plot_spectrum_signal.emit(self.df, self.column_names, True, x_min_val, x_max_val, y_min_val, y_max_val, self.is_ai_digitized)

    def load_spectrum_txt(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(caption="Open Text File", filter="Text Files (*.txt)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1, names=self.column_names)
                self.is_ai_digitized = False
                logger.debug(f"Загружен DataFrame:\n{self.df.head()}")
                logger.debug(f"Названия столбцов: {self.df.columns.tolist()}")
                if self.df.empty:
                    logger.error("Загруженный текстовый файл пустой.")
                    return None
                if not all(col in self.df.columns for col in self.column_names):
                    logger.error(f"Ожидаемые столбцы {self.column_names} не найдены в DataFrame. Имеются: {self.df.columns.tolist()}")
                    return None
                self._save_to_db(file_path)
                self.spectrum_loaded_signal.emit(self.df)
                self.plot_spectrum()
                return self.df
            except Exception as e:
                logger.warning(f"Ошибка при загрузке файла: {e}")
                return None

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName()
        if file_path:
            try:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                clr = cv2.imread(file_path)
                if image is None or clr is None:
                    logger.error(f"Не удалось загрузить изображение: {file_path}")
                    return None
                self.df = img2df(image, clr, 175, 5, 100, 57, 170, 220, 180, 220, 5)
                self.is_ai_digitized = False
                if self.df is None or self.df.empty:
                    logger.error("DataFrame из изображения пустой или не создан.")
                    return None
                self._save_to_db(file_path)
                self.spectrum_loaded_signal.emit(self.df)
                self.plot_spectrum()
                logger.debug(f"DataFrame из изображения: {self.df.head()}")
                return self.df
            except Exception as e:
                logger.warning(f"Ошибка при загрузке изображения: {e}")
                return None

    def load_edit_image(self, img):
        if img is None:
            logger.error("Переданное изображение None")
            return
        try:
            self.df = graph2df(img)
            self.is_ai_digitized = False
            if self.df is None or self.df.empty:
                logger.error("DataFrame из отредактированного изображения пустой или не создан.")
                return
            self._save_to_db("edited_image")
            self.spectrum_loaded_signal.emit(self.df)
            self.plot_spectrum()
            logger.debug(f"DataFrame из отредактированного изображения: {self.df.head()}")
        except Exception as e:
            logger.warning(f"Ошибка при обработке отредактированного изображения: {e}")

    # В метод load_extracted_data добавляем параметр save_to_db
    def load_extracted_data(self, df, x_min_val, x_max_val, y_min_val, y_max_val, 
                        is_descending, save_to_db=True, is_ai_digitized=False):
        if df is None or df.empty or "Длина_волны" not in df or "Интенсивность" not in df:
            logger.error("Получен пустой DataFrame или отсутствуют столбцы 'Длина_волны'/'Интенсивность'")
            return
        self.df = df
        self.is_ai_digitized = is_ai_digitized  # Сохраняем флаг
        logger.debug(f"Загружен DataFrame в load_extracted_data:\n{self.df.head()}")
        logger.debug(f"Интенсивность: min={self.df['Интенсивность'].min()}, max={self.df['Интенсивность'].max()}, is_ai_digitized={is_ai_digitized}")
        
        if save_to_db:
            self._save_to_db("extracted_data")
        
        self.spectrum_loaded_signal.emit(self.df)
        self.plot_spectrum_signal.emit(self.df, self.column_names, is_descending, x_min_val, x_max_val, y_min_val, y_max_val, is_ai_digitized)

    def _save_to_db(self, file_path):
        if self.df is None or self.df.empty:
            logger.error("DataFrame пустой. Сохранение в базу данных невозможно.")
            return
        try:
            image_name = os.path.basename(file_path) if file_path not in ["edited_image", "extracted_data"] else file_path
            wavelength = self.df['Длина_волны'].tolist()
            intensity = self.df['Интенсивность'].tolist()
            if not intensity:
                logger.error("Массив интенсивности пустой. Сохранение невозможно.")
                return
            min_idx = np.argmin(intensity)
            max_idx = np.argmax(intensity)
            min_peak = [wavelength[min_idx], intensity[min_idx]]
            max_peak = [wavelength[max_idx], intensity[max_idx]]
            self.db.save_sample(image_name, wavelength, intensity, min_peak, max_peak)
            logger.info(f"Данные сохранены в базу данных для {image_name}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении в базу данных: {e}")

    @pyqtSlot(tuple)
    def subctract_slice_background(self, slice_borders: tuple):
        if self.df is None or self.df.empty:
            logger.error("DataFrame пустой. Вычитание фона невозможно.")
            return
        logger.debug(f"Получены новые точки диапазона интегрирования: {slice_borders}")
        lower_bound, upper_bound = sorted(slice_borders)
        filtered_df = self.df[(self.df["Длина_волны"] >= lower_bound) & (self.df["Длина_волны"] <= upper_bound)].copy()
        if filtered_df.empty:
            logger.error("Фильтрованный DataFrame пустой. Вычитание фона невозможно.")
            return

        # Вычисляем линейный фон
        start_point = filtered_df.iloc[0]
        end_point = filtered_df.iloc[-1]
        m = ((end_point["Интенсивность"] - start_point["Интенсивность"])
             / (end_point["Длина_волны"] - start_point["Длина_волны"]))
        c = start_point["Интенсивность"] - m * start_point["Длина_волны"]

        # Вычитаем линейный фон
        filtered_df["Интенсивность"] = filtered_df["Интенсивность"] - (m * filtered_df["Длина_волны"] + c)

        # Находим максимальное значение в исходных данных в выбранном диапазоне
        original_max = self.df[(self.df["Длина_волны"] >= lower_bound) & (self.df["Длина_волны"] <= upper_bound)]["Интенсивность"].max()

        # Находим максимальное значение после вычитания фона
        current_max = filtered_df["Интенсивность"].max()

        # Вычисляем смещение, чтобы поднять данные до исходного максимума
        offset = original_max - current_max if current_max is not None else original_max

        # Применяем смещение
        filtered_df["Интенсивность"] = filtered_df["Интенсивность"] + offset

        # Определяем новые пределы для графика
        y_min_val = filtered_df["Интенсивность"].min()
        y_max_val = filtered_df["Интенсивность"].max()

        # Отправляем сигнал для построения графика
        self.plot_spectrum_signal.emit(filtered_df, self.column_names, False, lower_bound, upper_bound, y_min_val, y_max_val, self.is_ai_digitized)
        logger.debug(f"Вычтен фон, новый DataFrame: {filtered_df.head()}")

    def get_all_data(self):
        return self.db.get_all_data()

    def close_db(self):
        self.db.close()