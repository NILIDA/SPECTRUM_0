import numpy as np
import pandas as pd
from PyQt6.QtCore import pyqtSlot
from logger_config import logger

class GaussActionCallbacks:
    def __init__(self, graphical_area):
        self.graphical_area = graphical_area
        self.gaussian_params = graphical_area.gaussian_params
        self.current_range = None
        self.background_subtracted = False
        self.x_data = None
        self.y_data = None
        self.gaussian_drawn = False
        self.press_x = None

    def reset_gaussian_params(self):
        """Сбрасывает параметры гауссовых кривых и перерисовывает график."""
        self.gaussian_params.reset_df()
        logger.debug(f"Данные gaussian_params сброшены: {self.gaussian_params}")
        self.draw_gaussian_curves()

    def set_background_subtracted_range(self, x_range, y_data):
        """Устанавливает диапазон с вычтенным фоном."""
        self.current_range = x_range
        self.background_subtracted = True
        
        # Создаём x_data в убывающем порядке
        self.x_data = np.linspace(max(x_range), min(x_range), len(y_data))
        self.y_data = y_data
        
        logger.info(f"Установлен диапазон для гауссовых кривых: {x_range}")
        logger.debug(f"Размеры данных: x_data={len(self.x_data)}, y_data={len(self.y_data)}")

    def on_press(self, event):
        """Обработчик нажатия мыши."""
        if not self.background_subtracted:
            logger.warning("Сначала выделите диапазон и вычтите фон!")
            return
            
        if event.inaxes != self.graphical_area.ax:
            return

        self.graphical_area.mouse_pressed = True
        self.press_x = event.xdata
        self.gaussian_drawn = False

    def on_move(self, event):
        """Обработчик движения мыши."""
        if not self.background_subtracted:
            return

        if not self.graphical_area.mouse_pressed or event.inaxes != self.graphical_area.ax:
            return

        width = 2 * abs(self.press_x - event.xdata)
        if width < 1e-5:
            width = 1e-5

        # Используем убывающий x_data
        x = self.x_data
        y = self.gaussian(x, event.ydata, self.press_x, width)

        if self.gaussian_drawn:
            self.graphical_area.ax.lines[-1].set_ydata(y)
        else:
            self.graphical_area.ax.plot(x, y, 'r-', alpha=0.7)
            self.gaussian_drawn = True

        self.graphical_area.canvas.draw()

    def on_release(self, event):
        """Обработчик отпускания мыши."""
        if not self.background_subtracted:
            return

        self.graphical_area.mouse_pressed = False

        if not self.gaussian_drawn or event.ydata is None or event.xdata is None:
            return

        width = 2 * abs(self.press_x - event.xdata)
        if width < 1e-5:
            width = 1e-5

        new_row = pd.DataFrame({
            'Height': [max(event.ydata, 0)],
            'Position': [self.press_x],
            'Width': [width]
        })
        self.gaussian_params.concat_new_gaussian(new_row)
        logger.debug(f"Добавлена гауссова кривая: высота={event.ydata}, позиция={self.press_x}, ширина={width}")

        self.draw_gaussian_curves()
        self.gaussian_drawn = False

    def draw_gaussian_curves(self):
        """Рисует исходные данные, гауссовы кривые и их сумму."""
        if not self.background_subtracted:
            logger.warning("Не установлен диапазон с вычтенным фоном. Рисование гауссовых кривых пропущено.")
            return

        try:
            # Очищаем график
            self.graphical_area.ax.clear()
            
            # Проверяем, что x_data убывающий
            if not np.all(self.x_data[:-1] >= self.x_data[1:]):
                logger.error("x_data не в убывающем порядке! Переворачиваем x_data и y_data.")
                self.x_data = self.x_data[::-1]
                self.y_data = self.y_data[::-1]

            # Для эффекта "повторного отзеркаливания" переворачиваем y_data при отрисовке
            mirrored_y_data = self.y_data[::-1]
            
            # Рисуем исходные данные с убывающим x_data и отзеркаленными y_data
            self.graphical_area.ax.plot(self.x_data, mirrored_y_data, 'b-')
            
            # Вычисляем общую площадь всех пиков
            total_area = 0
            peak_areas = []
            
            # Инициализируем суммарную кривую нулями
            cumulative_y = np.zeros_like(self.y_data)

            # Рисуем каждую гауссову кривую
            colors = []
            for idx, row in self.gaussian_params.gaussian_df.iterrows():
                # Вычисляем гауссову кривую
                gauss_y = self.gaussian(self.x_data, row['Height'], row['Position'], row['Width'])
                
                # Вычисляем площадь под кривой
                area = np.trapz(gauss_y, self.x_data)
                peak_areas.append(area)
                total_area += abs(area)
                
                # Рисуем кривую
                line, = self.graphical_area.ax.plot(self.x_data, gauss_y, '--', alpha=0.7)
                color = line.get_color()
                colors.append(color)
                
                # Добавляем к суммарной кривой
                cumulative_y += gauss_y

                # Подписываем пик - положение и площадь
                self.graphical_area.ax.text(
                    row['Position'], 
                    row['Height'] * 0.9, 
                    f"Пик: {row['Position']:.1f}\nПлощадь: {abs(area):.2f}",
                    ha='center', va='top', fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=color, alpha=0.7)
                )

            # Рисуем сумму гауссовых кривых
            self.graphical_area.ax.plot(self.x_data, cumulative_y, 'k-')

            # Подписываем общую площадь
            self.graphical_area.ax.text(
                0.95, 0.95, 
                f"Общая площадь пиков: {total_area:.2f}",
                transform=self.graphical_area.ax.transAxes,
                fontsize=9, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.7)
            )

            # Настраиваем легенду и сетку
            self.graphical_area.ax.legend()
            self.graphical_area.ax.grid(True)

            # Устанавливаем убывающие пределы оси X
            self.graphical_area.ax.set_xlim(max(self.x_data), min(self.x_data))

            # Перерисовываем
            self.graphical_area.canvas.draw()

        except Exception as e:
            logger.error(f"Ошибка при перерисовке гауссовых кривых: {e}")
            self.graphical_area.ax.clear()
            self.graphical_area.canvas.draw()

    def gaussian(self, x: np.ndarray, height: float, center: float, width: float) -> np.ndarray:
        """Вычисляет гауссову кривую.
        
        Args:
            x: массив значений по оси X.
            height: высота пика.
            center: положение центра пика.
            width: ширина пика (полная ширина на половине высоты, FWHM).
            
        Returns:
            Массив значений гауссовой кривой.
        """
        # Преобразуем FWHM в sigma
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        # Гауссова функция
        return height * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))