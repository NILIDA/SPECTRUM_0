# train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import albumentations as A
from logger_config import logger

def unet_model(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    
    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    
    # Decoder
    u5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c4)
    u5 = layers.Concatenate()([u5, c3])
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)
    
    u6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c5)
    u6 = layers.Concatenate()([u6, c2])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c6)
    u7 = layers.Concatenate()([u7, c1])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = models.Model(inputs, outputs)
    return model

def augment_data(image, mask):
    aug = A.Compose([
        A.Rotate(limit=10, p=0.5),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.GaussNoise(p=0.3),
        # Добавляем Resize, чтобы гарантировать размер 256x256 после аугментации
        A.Resize(height=256, width=256, always_apply=True)
    ])
    augmented = aug(image=image, mask=mask)
    return augmented['image'], augmented['mask']

def load_data(data_dir):
    images = []
    masks = []
    img_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    expected_shape = (256, 256)
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        if os.path.exists(mask_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                logger.warning(f"Не удалось загрузить изображение или маску: {img_name}")
                continue
            # Масштабирование оригинальных изображений
            img = cv2.resize(img, expected_shape)
            mask = cv2.resize(mask, expected_shape)
            # Убедимся, что данные в формате uint8
            img = img.astype(np.uint8)
            mask = mask.astype(np.uint8)
            logger.debug(f"Тип данных до аугментации: img={img.dtype}, mask={mask.dtype}, shape={img.shape}")
            # Проверка размеров оригинальных данных
            if img.shape != expected_shape or mask.shape != expected_shape:
                logger.warning(f"Некорректный размер для {img_name}: img={img.shape}, mask={mask.shape}")
                continue
            # Аугментация
            aug_img, aug_mask = augment_data(img, mask)
            # Проверка размеров аугментированных данных
            if aug_img.shape != expected_shape or aug_mask.shape != expected_shape:
                logger.warning(f"Некорректный размер после аугментации для {img_name}: aug_img={aug_img.shape}, aug_mask={aug_mask.shape}")
                continue
            # Нормализация после аугментации
            images.append(img / 255.0)
            masks.append(mask / 255.0)
            images.append(aug_img / 255.0)
            masks.append(aug_mask / 255.0)
        else:
            logger.warning(f"Маска для изображения {img_name} не найдена")
    if not images:
        logger.error("Датасет пустой. Проверьте папку с данными")
        return np.array([]), np.array([])
    try:
        X = np.array(images)[..., np.newaxis]
        y = np.array(masks)[..., np.newaxis]
        logger.info(f"Загружено {len(X)} изображений с формой {X.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Ошибка при создании массивов: {e}")
        return np.array([]), np.array([])

def train_model(data_dir, epochs=20, batch_size=8):
    # Загрузка данных
    X, y = load_data(data_dir)
    if len(X) == 0:
        logger.error("Датасет пустой. Обучение невозможно.")
        return
    
    # Создание и компиляция модели
    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Обучение модели
    logger.info(f"Начало обучения модели. Количество изображений: {len(X)}")
    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    
    # Сохранение модели
    try:
        model.save('graph_line_detector.h5')
        logger.info("Модель сохранена как graph_line_detector.h5")
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели: {e}")

if __name__ == '__main__':
    data_dir = 'dataset'  # Убедитесь, что путь правильный
    train_model(data_dir)