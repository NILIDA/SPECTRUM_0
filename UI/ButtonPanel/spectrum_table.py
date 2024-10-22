from PyQt6.QtWidgets import QTableView
from PyQt6.QtCore import  pyqtSlot, Qt
from pandas_model import PandasModel
import pandas as pd
from logger_config import logger

class SpectrumTable(QTableView):
    def __init__(self, config, gaussian_params):
        super().__init__()
        self.config = config
        self.gaussian_params = gaussian_params
        self.setFixedHeight(300)        

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected_row()

    def delete_selected_row(self):
        logger.debug('Начало процесса удаления выбранной строки')
        selected_model = self.selectionModel()
        if selected_model.hasSelection():
            selected_rows = selected_model.selectedRows()
            if selected_rows:
                selected_row = selected_rows[0].row()
                logger.debug(f'Выбрана строка для удаления: {selected_row}')
                
                success = self.model().removeRow(selected_row)
                if success:
                    logger.debug('Строка успешно удалена из модели')
                else:
                    logger.debug('Не удалось удалить строку из модели')

                # Обновление исходного DataFrame
                updated_dataframe = self.model()._data
                logger.debug(f'Обновлённый DataFrame перед отправкой сигнала:\n{updated_dataframe.head()}')
                self.spectrum_model.data_changed_signal.emit(updated_dataframe.head())
                logger.debug('Сигнал об изменении данных отправлен')
            else:
                logger.debug('Нет выбранных строк для удаления')
        else:
            logger.debug('Нет выбранной модели или выборки')
        logger.debug('Завершение процесса удаления строки')
    
    @pyqtSlot(pd.DataFrame)
    def update_table(self, dataframe):
        logger.debug('Сигнал о новой таблице получен')
        self.spectrum_model = PandasModel(dataframe)
        if 'window_length' in dataframe.columns:
            self.spectrum_model.data_changed_signal.connect(
                self.config.update_config
            )
        elif 'Height' in dataframe.columns:
            self.spectrum_model.data_changed_signal.connect(
                self.gaussian_params.update_params
            )
        self.setModel(self.spectrum_model)