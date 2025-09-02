# UI/GraphicalArea/custom_toolbar.py
import pandas as pd
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from config import SpectrumConfig
from resources import resource_path
from logger_config import logger

class CustomToolbar(NavigationToolbar):
    home_signal = pyqtSignal()  # Сигнал для кнопки "Домой"
    restore_graphical_view_signal = pyqtSignal()  # Сигнал для кнопки "Окно графика"
    gauss_action_signal = pyqtSignal(pd.DataFrame)
    integral_action_signal = pyqtSignal(pd.DataFrame)
    actions_deactivated_signal = pyqtSignal()

    def __init__(self, canvas, graphical_area):
        super().__init__(canvas, graphical_area)
        self.config = SpectrumConfig()
        self.gaussian_params = graphical_area.gaussian_params
        self.create_actions()

    def create_actions(self):
        # Переопределяем кнопку "Домой"
        self._actions['home'].triggered.disconnect()  # Отключаем стандартное поведение
        self._actions['home'].triggered.connect(self.home_signal.emit)

        # Кнопка "Интегрирование"
        self.integral_action = self.add_action(resource_path('icons/integral_icon.png'), 'integral')
        
        # Кнопка "Гауссы"
        self.gauss_action = self.add_action(resource_path('icons/gauss_icon.png'), 'gauss')
        
        # Кнопка "Окно графика"
        self.graph_action = self.add_action(resource_path('icons/restore_df_plot_icon.png'), 'graph')

    def add_action(self, icon_path, action_name, checkable=True):
        action = QAction(QIcon(icon_path), '', self)
        action.setCheckable(checkable)
        action.triggered.connect(lambda: self.toggle_action(action_name))
        self.addAction(action)
        return action

    def toggle_action(self, action_name):
        if action_name == 'integral':
            self.activate_action(self.integral_action, [self.gauss_action])
            self.integral_action_signal.emit(self.config.Savitzky_df)
        elif action_name == 'gauss':
            self.activate_action(self.gauss_action, [self.integral_action])
            self.gauss_action_signal.emit(self.gaussian_params.gaussian_df)
        elif action_name == 'graph':
            self.deactivate_all_actions()
            self.restore_graphical_view_signal.emit()  # Восстановить график и таблицу

    def activate_action(self, active_action, other_actions):
        active_action.setChecked(True)
        for action in other_actions:
            action.setChecked(False)

    def deactivate_all_actions(self):
        self.integral_action.setChecked(False)
        self.gauss_action.setChecked(False)
        self.actions_deactivated_signal.emit()