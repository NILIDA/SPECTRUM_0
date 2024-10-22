import numpy as np

class IntegralActionCallbacks:
    def __init__(self, graphical_area):
        self.graphical_area = graphical_area
        
    def on_press(self, event):
        self.graphical_area.mouse_pressed = True
        self.graphical_area.press_x = event.xdata
        self.graphical_area.original_xlim = self.graphical_area.ax.get_xlim()
        self.graphical_area.ax.set_facecolor('white')
        self.graphical_area.canvas.draw()

    def on_move(self, event):
        for region in self.graphical_area.shading_regions:
                region.remove()
        self.graphical_area.shading_regions.clear()
        self.graphical_area.ax.set_xlim(self.graphical_area.original_xlim)
        # Создание затемненных областей в зависимости от положения мыши
        if event.xdata < self.graphical_area.press_x:
            region1 = self.graphical_area.ax.axvspan(self.graphical_area.ax.get_xlim()[0], event.xdata, color='gray', alpha=0.5)
            region2 = self.graphical_area.ax.axvspan(self.graphical_area.press_x, self.graphical_area.ax.get_xlim()[1], color='gray', alpha=0.5)
        else:
            region1 = self.graphical_area.ax.axvspan(self.graphical_area.ax.get_xlim()[0], self.graphical_area.press_x, color='gray', alpha=0.5)
            region2 = self.graphical_area.ax.axvspan(event.xdata, self.graphical_area.ax.get_xlim()[1], color='gray', alpha=0.5)
        self.graphical_area.shading_regions.extend([region1, region2])
        self.graphical_area.canvas.draw()        

    def on_release(self, event):
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
        # Определяем начальную и конечную точки для интегрирования
        start_x = min(self.graphical_area.press_x, event.xdata)
        end_x = max(self.graphical_area.press_x, event.xdata)            
        # Выбираем данные для интегрирования
        mask = (self.graphical_area.x_data >= start_x) & (self.graphical_area.x_data <= end_x)
        x_selected = self.graphical_area.x_data[mask]
        y_selected = self.graphical_area.y_data[mask]
        # Вычисляем интеграл
        integral_value = np.trapz(y_selected, x_selected)
        # Находим минимальное значение y и соответствующее значение x
        min_index = np.argmin(y_selected)
        min_x = x_selected.iloc[min_index]  # Использование iloc для доступа по позиции
        # Отображаем значение интеграла и минимального значения x на графике
        self.graphical_area.ax.text(0.95, 0.05, f'Площадь: {integral_value:.2f}\nМинимум: {min_x:.2f}',
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=self.graphical_area.ax.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        # Перерисовываем холст
        self.graphical_area.canvas.draw()
