# pandas_model.py
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtCore import pyqtSignal
from logger_config import logger
import pandas as pd

class PandasModel(QAbstractTableModel):
    data_changed_signal = pyqtSignal(pd.DataFrame)

    def __init__(self, data, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.UserRole:
                return self._data.iloc[index.row(), index.column()]
                
            if role == Qt.ItemDataRole.DisplayRole:
                value = self._data.iloc[index.row(), index.column()]
                # Для виджетов возвращаем пустую строку
                return "" if hasattr(value, 'setParent') else str(value)
        return None

    def removeRow(self, row, parent=None):
        self.beginRemoveRows(QModelIndex(), row, row)
        try:
            self._data = self._data.drop(self._data.index[row])
            self.endRemoveRows()
            self.data_changed_signal.emit(self._data)
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении строки: {e}")
            self.endRemoveRows()
            return False

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._data.columns[section]
        if orientation == Qt.Orientation.Vertical:
            return self._data.index[section]    

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        row = index.row()
        col = index.column()
        column_name = self._data.columns[col]
        
        # Разрешаем редактирование только для определенных столбцов
        editable_columns = ['image_name']  # Можно добавить другие столбцы, если нужно
        if column_name not in editable_columns:
            return False
            
        try:
            self._data.iat[row, col] = value  # Сохраняем как строку
            self.dataChanged.emit(index, index, [role])
            self.data_changed_signal.emit(self._data)
            logger.debug(f"Данные изменены в строке {row}, столбце {col}: {value}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при установке данных: {e}")
            return False

    def flags(self, index):
        column_name = self._data.columns[index.column()]
        if column_name == 'image_name':  # Только image_name редактируемый
            return Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable