# Импортируем необходимые библиотеки
import cv2
import numpy as np
import pandas as pd
from cv2 import GaussianBlur
from cv2 import boxFilter


def img2df(img, color, block_size, c, l_b, l_g, l_r, u_b, u_g, u_r, blur):
    processed_points = np.empty((0, 2), dtype=int)
    blured = cv2.GaussianBlur(img, (blur, blur), 1, )
    def bersen_thresholding(image, block_size, c):
        # Применяем адаптивный пороговый фильтр Берсена
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        return binary_image

    # Применение метода бинаризации по Берсену
    binary_bersen = bersen_thresholding(blured, block_size, c)
    # Алгоритм Собеля
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    gradient_x = cv2.filter2D(binary_bersen, cv2.CV_64F, sobel_x)
    gradient_y = cv2.filter2D(binary_bersen, cv2.CV_64F, sobel_y)

    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Бинаризация
    threshold = np.mean(gradient)
    binary_gradient = np.where(gradient > threshold, 255, 0).astype(np.uint8)  # Преобразование типа данных

    # Поиск контуров
    contours, _ = cv2.findContours(binary_gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Нахождение прямоугольника с максимальной площадью
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Выравнивание и вырезание области
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)

    # Вычисление матрицы поворота
    angle = rect[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    M = cv2.getRotationMatrix2D((rect[0][0], rect[0][1]), angle, 1.0)

    # Поворот изображения
    rotated = cv2.warpAffine(color, M, (img.shape[1], img.shape[0]))

    # Вырезание области
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped = rotated[y:y + h, x:x + w]
    # Цветовая сегментация
    # Определение диапазона цветов для темно-синего (в формате BGR)
    lower_blue = np.array([l_b, l_g, l_r])
    upper_blue = np.array([u_b, u_g, u_r])

    # Преобразование изображения в цветовое пространство BGR
    image_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

    # Создание маски, соответствующей области темно-синего цвета
    mask = cv2.inRange(image_bgr, lower_blue, upper_blue)

    # Применение маски к исходному изображению
    result = cv2.bitwise_and(cropped, cropped, mask=mask)
    # Бинаризация
    binary_image = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)[1]
    # Поиск координат белых точек
    white_points = np.column_stack(np.where(binary_image == 255))
    # Преобразование координат
    width = binary_image.shape[1]
    height = binary_image.shape[0]

    # Выполнение преобразования
    white_points_transformed = np.copy(white_points)  # Создание копии массива, чтобы не изменить исходный
    white_points_transformed[:, 1] = -4000 * white_points[:, 1] / width + 4000
    white_points_transformed[:, 0] = -(100 * white_points[:, 0] / height) + 97

    unique_x = np.unique(white_points_transformed[:, 1])
    neighborhood = 5
    for x in unique_x:
        # Выбираем точки в окрестности заданной координаты x
        neighborhood_points = white_points_transformed[np.abs(white_points_transformed[:, 1] - x) <= neighborhood]

        # Если в окрестности есть точки
        if len(neighborhood_points) > 0:
            # Вычисляем среднее значение y в этой окрестности
            mean_y = np.mean(neighborhood_points[:, 1])

            # Фильтрация точек в этой окрестности
            filtered_points = neighborhood_points[np.abs(neighborhood_points[:, 1] - mean_y) <= 2500]

            # Добавляем уникальные точки в итоговый массив
            unique_filtered_points = np.unique(filtered_points, axis=1)
            y = np.mean(unique_filtered_points[:, 1])
            e_processed_points = np.array([x, y])
            processed_points = np.vstack((processed_points, e_processed_points))
    data1 = {
         'Длина_волны': processed_points[:, 0],
         'Интенсивность': processed_points[:, 1]
    }

    df = pd.DataFrame(data=data1)
    df1 = df.sort_values(by='Длина_волны', ascending=True )
    return df1

def graph2df(result):
    # Бинаризация
    binary_image = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)[1]
    # Поиск координат белых точек
    white_points = np.column_stack(np.where(binary_image == 255))
    # Преобразование координат
    width = binary_image.shape[1]
    height = binary_image.shape[0]

    # Выполнение преобразования
    white_points_transformed = np.copy(white_points)  # Создание копии массива, чтобы не изменить исходный
    white_points_transformed[:, 1] = -4000 * white_points[:, 1] / width + 4000
    white_points_transformed[:, 0] = -(100 * white_points[:, 0] / height) + 97

    unique_x = np.unique(white_points_transformed[:, 1])
    neighborhood = 5
    for x in unique_x:
        # Выбираем точки в окрестности заданной координаты x
        neighborhood_points = white_points_transformed[np.abs(white_points_transformed[:, 1] - x) <= neighborhood]

        # Если в окрестности есть точки
        if len(neighborhood_points) > 0:
            # Вычисляем среднее значение y в этой окрестности
            mean_y = np.mean(neighborhood_points[:, 1])

            # Фильтрация точек в этой окрестности
            filtered_points = neighborhood_points[np.abs(neighborhood_points[:, 1] - mean_y) <= 2500]

            # Добавляем уникальные точки в итоговый массив
            unique_filtered_points = np.unique(filtered_points, axis=1)
            y = np.mean(unique_filtered_points[:, 1])
            e_processed_points = np.array([x, y])
            processed_points = np.vstack((processed_points, e_processed_points))
    data1 = {
         'Длина_волны': processed_points[:, 0],
         'Интенсивность': processed_points[:, 1]
    }

    df = pd.DataFrame(data=data1)
    df1 = df.sort_values(by='Длина_волны', ascending=True )
    return df1


def modify(img, color, block_size, c, l_b, l_g, l_r, u_b, u_g, u_r, blur):
    processed_points = np.empty((0, 2), dtype=int)
    blured = cv2.GaussianBlur(img, (blur, blur), 1, )
    def bersen_thresholding(image, block_size, c):
        # Применяем адаптивный пороговый фильтр Берсена
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        return binary_image

    # Применение метода бинаризации по Берсену
    binary_bersen = bersen_thresholding(blured, block_size, c)
    # Алгоритм Собеля
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    gradient_x = cv2.filter2D(binary_bersen, cv2.CV_64F, sobel_x)
    gradient_y = cv2.filter2D(binary_bersen, cv2.CV_64F, sobel_y)

    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Бинаризация
    threshold = np.mean(gradient)
    binary_gradient = np.where(gradient > threshold, 255, 0).astype(np.uint8)  # Преобразование типа данных

    # Поиск контуров
    contours, _ = cv2.findContours(binary_gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Нахождение прямоугольника с максимальной площадью
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Выравнивание и вырезание области
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)

    # Вычисление матрицы поворота
    angle = rect[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    M = cv2.getRotationMatrix2D((rect[0][0], rect[0][1]), angle, 1.0)

    # Поворот изображения
    rotated = cv2.warpAffine(color, M, (img.shape[1], img.shape[0]))

    # Вырезание области
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped = rotated[y:y + h, x:x + w]
    # Цветовая сегментация
    # Определение диапазона цветов для темно-синего (в формате BGR)
    lower_blue = np.array([l_b, l_g, l_r])
    upper_blue = np.array([u_b, u_g, u_r])

    # Преобразование изображения в цветовое пространство BGR
    image_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

    # Создание маски, соответствующей области темно-синего цвета
    mask = cv2.inRange(image_bgr, lower_blue, upper_blue)

    # Применение маски к исходному изображению
    result = cv2.bitwise_and(cropped, cropped, mask=mask)
    # Бинаризация
    binary_image_2 = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)[1]

    return binary_image_2