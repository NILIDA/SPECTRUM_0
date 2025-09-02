# db_window.py (полный исправленный код)
from PyQt6.QtWidgets import (QMainWindow, QTableView, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QHeaderView, QLabel, QComboBox,
                            QLineEdit, QFormLayout, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt
from pandas_model import PandasModel
from logger_config import logger
import pandas as pd
from PyQt6.QtWidgets import QFileDialog

class DBWindow(QMainWindow):
    def __init__(self, spectrum_data_frame):
        super().__init__()
        self.setWindowTitle("База данных графиков")
        self.setMinimumSize(800, 600)
        self.spectrum_data_frame = spectrum_data_frame

        # Параметры пагинации
        self.rows_per_page = 10
        self.current_page = 1
        self.total_pages = 0
        self.current_data = pd.DataFrame()

        # Основной виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Таблица для отображения данных
        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table_view.setSortingEnabled(True)
        self.table_view.clicked.connect(self.on_row_clicked)
        layout.addWidget(self.table_view)

        # Панель пагинации
        self.pagination_widget = QWidget()
        self.pagination_layout = QHBoxLayout(self.pagination_widget)
        
        # Метка текущей страницы
        self.page_label = QLabel("Страница 1 из 1")
        self.pagination_layout.addWidget(self.page_label)

        # Кнопки навигации
        self.prev_button = QPushButton("Назад")
        self.next_button = QPushButton("Вперёд")
        self.prev_button.clicked.connect(self.prev_page)
        self.next_button.clicked.connect(self.next_page)
        self.pagination_layout.addWidget(self.prev_button)
        self.pagination_layout.addWidget(self.next_button)

        # Выпадающий список для выбора страницы
        self.page_selector = QComboBox()
        self.page_selector.currentIndexChanged.connect(self.jump_to_page)
        self.pagination_layout.addWidget(QLabel("Перейти к странице:"))
        self.pagination_layout.addWidget(self.page_selector)

        layout.addWidget(self.pagination_widget)

        # Кнопка удаления
        self.delete_button = QPushButton("Удалить выбранный пример")
        self.delete_button.clicked.connect(self.delete_selected_sample)
        layout.addWidget(self.delete_button)

        # Группа действий с экспериментом
        actions_group = QGroupBox("Действия с экспериментом")
        actions_layout = QFormLayout(actions_group)
        
        # Поле для ввода индекса
        self.index_input = QLineEdit()
        self.index_input.setPlaceholderText("Введите ID эксперимента")
        actions_layout.addRow(QLabel("ID эксперимента:"), self.index_input)
        
        # Кнопки действий
        buttons_layout = QHBoxLayout()
        
        self.plot_button = QPushButton("Построить график")
        self.plot_button.clicked.connect(self.plot_selected_spectrum)
        buttons_layout.addWidget(self.plot_button)
        
        self.export_button = QPushButton("Экспорт в TXT")
        self.export_button.clicked.connect(self.export_selected_spectrum)
        buttons_layout.addWidget(self.export_button)
        
        actions_layout.addRow(buttons_layout)
        
        layout.addWidget(actions_group)

        # Инициализация таблицы
        self.update_table()

    def on_row_clicked(self, index):
        """Автоматически заполняет поле ID при выборе строки в таблице."""
        if index.isValid():
            record_id = self.model._data.iloc[index.row()]['id']
            self.index_input.setText(str(record_id))

    def update_table(self):
        """Обновление таблицы с данными для текущей страницы."""
        logger.debug(f"Обновление таблицы для страницы {self.current_page}")
        
        # Получение данных для текущей страницы
        self.current_data = self.spectrum_data_frame.db.get_page_data(self.current_page, self.rows_per_page)
        
        # Подсчёт общего числа записей для пагинации
        total_records = self.spectrum_data_frame.db.count_total_records()
        self.total_pages = max(1, (total_records + self.rows_per_page - 1) // self.rows_per_page)
        self.current_page = min(self.current_page, self.total_pages)

        if self.current_data.empty:
            logger.debug("Нет данных для отображения на странице")
            self.model = PandasModel(pd.DataFrame(columns=['id', 'image_name', 'wavelength', 'intensity', 'min_peak', 'max_peak']))
        else:
            logger.debug(f"Получено {len(self.current_data)} записей для страницы {self.current_page}")
            # Создаем копию для отображения (преобразуем списки в строки)
            display_data = self.current_data.copy()
            display_data['wavelength'] = display_data['wavelength'].apply(str)
            display_data['intensity'] = display_data['intensity'].apply(str)
            display_data['min_peak'] = display_data['min_peak'].apply(str)
            display_data['max_peak'] = display_data['max_peak'].apply(str)
            
            self.model = PandasModel(display_data)

        # Установка модели
        self.model.data_changed_signal.connect(self.update_db)
        self.table_view.setUpdatesEnabled(False)
        self.table_view.setModel(self.model)
        self.table_view.setUpdatesEnabled(True)

        # Настройка заголовков
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Обновление пагинации
        self.page_label.setText(f"Страница {self.current_page} из {self.total_pages}")
        self.prev_button.setEnabled(self.current_page > 1)
        self.next_button.setEnabled(self.current_page < self.total_pages)

        # Обновление выпадающего списка
        self.page_selector.blockSignals(True)
        self.page_selector.clear()
        self.page_selector.addItems([str(i) for i in range(1, self.total_pages + 1)])
        self.page_selector.setCurrentIndex(self.current_page - 1)
        self.page_selector.blockSignals(False)

        logger.debug(f"Таблица обновлена: страница {self.current_page}, строк {self.model.rowCount()}")

    def get_selected_record(self):
        """Возвращает выбранную запись по ID."""
        try:
            record_id = int(self.index_input.text().strip())
            
            # Ищем запись в текущей странице
            if not self.current_data.empty:
                record = self.current_data[self.current_data['id'] == record_id]
                if not record.empty:
                    return record.iloc[0].to_dict()
            
            # Если не нашли на текущей странице, запрашиваем из БД
            return self.spectrum_data_frame.db.get_record_by_id(record_id)
            
        except (ValueError, TypeError):
            QMessageBox.warning(self, "Ошибка", "Введите корректный ID эксперимента")
            return None

    def plot_selected_spectrum(self):
        """Построение графика для выбранной записи."""
        record = self.get_selected_record()
        if record is None:
            return
            
        # Создаем DataFrame из данных записи
        try:
            # Получаем списки значений
            wavelengths = record['wavelength']
            intensities = record['intensity']
            
            df = pd.DataFrame({
                'Длина_волны': wavelengths,
                'Интенсивность': intensities
            })
            
            # Определяем пределы осей
            x_min = min(wavelengths)
            x_max = max(wavelengths)
            y_min = min(intensities)
            y_max = max(intensities)
            
            # Отправляем данные для построения графика
            self.spectrum_data_frame.load_extracted_data(
                df, 
                x_min, x_max, y_min, y_max,
                is_descending=True,
                save_to_db=False
            )
            self.close()
            
        except Exception as e:
            logger.error(f"Ошибка при построении графика: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось построить график: {str(e)}")

    def export_selected_spectrum(self):
        """Экспорт данных спектра в TXT-файл."""
        record = self.get_selected_record()
        if record is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Сохранить спектр как TXT", 
            f"{record['image_name']}.txt", 
            "Text Files (*.txt)"
        )
        
        if not file_path:
            return
            
        if not file_path.endswith('.txt'):
            file_path += '.txt'
            
        try:
            # Получаем списки значений
            wavelengths = record['wavelength']
            intensities = record['intensity']
            
            with open(file_path, 'w') as f:
                # Записываем данные построчно
                for wl, intens in zip(wavelengths, intensities):
                    f.write(f"{wl}\t{intens}\n")
            
            QMessageBox.information(self, "Успех", f"Данные успешно экспортированы в:\n{file_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте данных: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать данные: {str(e)}")

    def prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.update_table()

    def next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.update_table()

    def jump_to_page(self):
        try:
            self.current_page = int(self.page_selector.currentText())
            self.update_table()
        except ValueError:
            logger.error("Ошибка: Неверный номер страницы в селекторе")

    def update_db(self, dataframe):
        import ast
        logger.debug("Обновление данных в базе")
        for _, row in dataframe.iterrows():
            try:
                self.spectrum_data_frame.db.update_record(
                    row['id'],
                    image_name=row['image_name'],
                    wavelength=ast.literal_eval(row['wavelength']),
                    intensity=ast.literal_eval(row['intensity']),
                    min_peak=ast.literal_eval(row['min_peak']),
                    max_peak=ast.literal_eval(row['max_peak'])
                )
            except Exception as e:
                logger.error(f"Ошибка при обновлении записи id={row['id']}: {e}")
        self.update_table()

    def delete_selected_sample(self):
        selected = self.table_view.selectionModel().selectedRows()
        if selected:
            logger.debug(f"Выбрано {len(selected)} строк для удаления")
            deleted_ids = []
            for index in selected:
                record_id = int(self.model._data.iloc[index.row()]['id'])
                logger.debug(f"Удаление примера с id={record_id}")
                self.spectrum_data_frame.db.delete_by_id(record_id)
                deleted_ids.append(record_id)
            if deleted_ids:
                logger.debug(f"Удалены примеры с id: {deleted_ids}")
            else:
                logger.debug("Удаление не выполнено")
            self.update_table()
        else:
            logger.debug("Ничего не выбрано для удаления")

    def closeEvent(self, event):
        logger.info("Окно базы данных закрыто")
        event.accept()