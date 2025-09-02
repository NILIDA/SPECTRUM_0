# UI/GraphicalArea/integral_action_callbacks.py
import numpy as np
from logger_config import logger

class IntegralActionCallbacks:
    def __init__(self, graphical_area):
        self.graphical_area = graphical_area

    def on_press(self, event):
        if event.xdata is None:
            logger.debug("Нажатие мыши вне осей графика")
            return
        self.graphical_area.mouse_pressed = True
        self.graphical_area.press_x = event.xdata
        self.graphical_area.original_xlim = self.graphical_area.ax.get_xlim()
        self.graphical_area.ax.set_facecolor('white')
        # Сбрасываем масштаб до исходного
        self.graphical_area.ax.set_xlim(self.graphical_area.original_xlim)
        self.graphical_area.canvas.draw()

    def on_move(self, event):
        if not self.graphical_area.mouse_pressed or event.xdata is None or not event.inaxes:
            return
        # Очищаем предыдущие затемнённые области
        for region in self.graphical_area.shading_regions:
            region.remove()
        self.graphical_area.shading_regions.clear()

        # Получаем текущие пределы осей
        xlim = self.graphical_area.original_xlim
        x_min, x_max = min(xlim), max(xlim)

        # Определяем границы выделенной области
        start_x = min(self.graphical_area.press_x, event.xdata)
        end_x = max(self.graphical_area.press_x, event.xdata)

        # Создаём затемнённые области вне выделенного диапазона
        region1 = self.graphical_area.ax.axvspan(x_min, start_x, color='gray', alpha=0.5)
        region2 = self.graphical_area.ax.axvspan(end_x, x_max, color='gray', alpha=0.5)
        self.graphical_area.shading_regions.extend([region1, region2])

        logger.debug(f"Затемнённые области: [{x_min}, {start_x}] и [{end_x}, {x_max}]")
        self.graphical_area.canvas.draw()

    def on_release(self, event):
        if event.xdata is None or not event.inaxes:
            logger.debug("Отпускание мыши вне осей графика")
            return
        self.graphical_area.mouse_pressed = False
        for region in self.graphical_area.shading_regions:
            region.remove()
        self.graphical_area.shading_regions.clear()
        self.graphical_area.ax.set_xlim(self.graphical_area.original_xlim)
        self.graphical_area.canvas.draw()
        # Испускание сигнала с координатами
        if self.graphical_area.press_x is not None and event.xdata is not None:
            self.graphical_area.mouse_released_signal.emit((self.graphical_area.press_x, event.xdata))
            self.display_integral_value(event)

    def display_integral_value(self, event):
        # Проверка валидности входных данных
        if self.graphical_area.press_x is None or event.xdata is None:
            logger.error("Некорректные координаты для интегрирования")
            return

        # Определяем начальную и конечную точки для интегрирования
        start_x = min(self.graphical_area.press_x, event.xdata)
        end_x = max(self.graphical_area.press_x, event.xdata)
        logger.debug(f"Диапазон интегрирования: [{start_x}, {end_x}]")

        # Выбираем данные для интегрирования
        mask = (self.graphical_area.x_data >= start_x) & (self.graphical_area.x_data <= end_x)
        x_selected = self.graphical_area.x_data[mask]
        y_selected = self.graphical_area.y_data[mask]

        # Проверка на пустые данные
        if x_selected.size == 0 or y_selected.size == 0:
            logger.error(f"Нет данных в диапазоне [{start_x}, {end_x}]. Размер x_selected: {x_selected.size}, y_selected: {y_selected.size}")
            return

        # Преобразуем данные в numpy-массивы
        x_array = x_selected.to_numpy() if hasattr(x_selected, 'to_numpy') else np.array(x_selected)
        y_array = y_selected.to_numpy() if hasattr(y_selected, 'to_numpy') else np.array(y_selected)

        # Учитываем смещение для ИИ-оцифровки
        if self.graphical_area.is_ai_digitized:
            y_array = y_array
            logger.debug(f"После смещения для ИИ: y_array[:5]={y_array[:5]}, min={np.min(y_array)}, max={np.max(y_array)}")

        # Проверка на наличие NaN или некорректных значений
        if np.any(np.isnan(x_array)) or np.any(np.isnan(y_array)):
            logger.error("Обнаружены NaN в данных для интегрирования")
            return

        logger.debug(f"Выбрано точек: {len(x_array)}, x_array: {x_array[:5]}..., y_array: {y_array[:5]}...")

        # Сортировка данных по возрастанию для np.trapz
        sorted_indices = np.argsort(x_array)
        x_array = x_array[sorted_indices]
        y_array = y_array[sorted_indices]

        # Проверка на достаточное количество точек
        if len(x_array) < 2:
            logger.error("Недостаточно точек для интегрирования: требуется минимум 2 точки")
            return

        # Интерполяция для повышения точности
        x_dense = np.linspace(min(x_array), max(x_array), 1000)
        y_dense = np.interp(x_dense, x_array, y_array)

        # Вычисляем интеграл
        raw_integral = np.trapz(y_dense, x_dense)
        integral_value = np.abs(raw_integral)  # Гарантируем положительную площадь
        logger.debug(f"Интеграл: Сырой={raw_integral:.2f}, Исправленный={integral_value:.2f}")

        # Находим минимальное значение y в исходных данных и соответствующее значение x
        min_index = np.argmin(y_array)
        min_x = x_array[min_index]

        # Отображаем значение интеграла и минимального значения x на графике
        self.graphical_area.ax.text(
            0.95, 0.05, f'Площадь: {integral_value:.2f}\nМинимум: {min_x:.2f}',
            verticalalignment='bottom', horizontalalignment='right',
            transform=self.graphical_area.ax.transAxes, fontsize=8,
            bbox=dict(facecolor='white', alpha=0.5))

        # Добавляем заштриховку области интегрирования
        self.graphical_area.ax.fill_between(x_array, y_array, alpha=0.3, label="Интегрируемая область")
        self.graphical_area.ax.legend()

        # Передаем данные в гауссовы кривые
        self.graphical_area.gauss_callbacks.set_background_subtracted_range(
            (start_x, end_x), 
            y_dense
        )

        # Перерисовываем холст
        self.graphical_area.canvas.draw()