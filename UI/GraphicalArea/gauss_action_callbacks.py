import numpy as np
import pandas as pd
from PyQt6.QtCore import pyqtSlot
from logger_config import logger

class GaussActionCallbacks:
    def __init__(self, graphical_area):
        self.graphical_area = graphical_area
        self.gaussian_params = graphical_area.gaussian_params

    def reset_gaussian_params(self):
        self.gaussian_params.reset_df()
        logger.debug(f"Данные gaussian_params сброшены: {self.gaussian_params}")
    
    def on_press(self, event):
        self.graphical_area.mouse_pressed = True
        self.press_x = event.xdata        
        self.gaussian_drawn = False

    def on_move(self, event):
        if event.inaxes and self.graphical_area.mouse_pressed:
            w = 2 * abs(self.press_x - event.xdata)
            self.x_data = self.graphical_area.ax.lines[0].get_xdata()
            x = np.linspace(min(self.x_data), max(self.x_data), 1000)
            y = self.gaussian(x, event.ydata, self.press_x, w)
            if self.gaussian_drawn:
                self.graphical_area.ax.lines[-1].set_xdata(x)
                self.graphical_area.ax.lines[-1].set_ydata(y)
            else:
                self.graphical_area.ax.plot(x, y)
                self.gaussian_drawn = True
            self.graphical_area.canvas.draw()
        
    def on_release(self, event):
        self.graphical_area.mouse_pressed = False
        if self.gaussian_drawn:
            new_row = pd.DataFrame({'Height': [event.ydata],
                                    'Position': [self.press_x],
                                    'Width': [2 * abs(self.press_x - event.xdata)]})
            self.gaussian_params.concat_new_gaussian(new_row)
            logger.debug(
                f"Событие отпуска мыши: высота={event.ydata}, позиция={self.press_x}, ширина={2 * abs(self.press_x - event.xdata)}")
            self.draw_gaussian_curves()
            self.gaussian_drawn = False

    def draw_gaussian_curves(self):        
        self.graphical_area.ax.clear()
        self.graphical_area.ax.plot(self.graphical_area.x_data, self.graphical_area.y_data)
        # Список для хранения цветов кривых
        curve_colors = []
        # Отрисовка каждой гауссовской кривой и сохранение её цвета
        for index, row in self.gaussian_params.gaussian_df.iterrows():
            line, = self.graphical_area.ax.plot(
                self.x_data, self.gaussian(self.x_data, row['Height'], row['Position'], row['Width'])
            )
            curve_colors.append(line.get_color())
        # Отображение кумулятивной кривой
        self.graphical_area.ax.plot(self.x_data, self.cumulitive_gauss_func())
        # Добавляем значение интеграла и отображаем квадратик с цветом каждой функции
        for index, (idx, row) in enumerate(self.gaussian_params.gaussian_df.iterrows()):
            gauss_curve = self.gaussian(self.x_data, row['Height'], row['Position'], row['Width'])
            integral_value = np.trapz(gauss_curve, self.x_data)
            # Получаем цвет линии
            line_color = curve_colors[index]
            # Отрисовываем квадратик с цветом
            self.graphical_area.ax.scatter(
                [0.045], [0.25 - index * 0.03], color=line_color, transform=self.graphical_area.ax.transAxes, s=100)
            # Отображаем текст
            self.graphical_area.ax.text(
                0.065, 0.25 - index * 0.03, f"{row['Position']:.2f}, Площадь: {integral_value:.2f}",
                transform=self.graphical_area.ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        self.graphical_area.canvas.draw()
    
    def cumulitive_gauss_func(self):
        logger.debug(f"Данные gaussian_params в функции кумулятивной линии: {self.gaussian_params.gaussian_df}")        
        return self.gaussian_params.gaussian_df.apply(lambda x: self.gaussian(self.x_data, *x), axis=1).sum()
    
    @staticmethod
    def fit_function(x, *params):
        return GaussActionCallbacks.peaks(x, params)

    @staticmethod
    def peaks(x: np.array, params: tuple) -> np.array:
        y = np.zeros_like(x)
        i = 0
        h = params[3*i]
        z = params[3*i+1]
        w = params[3*i+2]            
        y = y + GaussActionCallbacks.gaussian(x, h, z, w)
        return y
    
    @staticmethod
    def gaussian(x: np.ndarray, h: float, z: float, w: float) -> np.ndarray:             
        return h * np.exp(-((x - z) ** 2) / (2 * w ** 2))
    
    
    