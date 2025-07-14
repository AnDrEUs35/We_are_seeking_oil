import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import rasterio # Импортируем rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

# --- 1. Конфигурация и пути к данным ---
# Вам нужно будет изменить эти пути в соответствии с вашей локальной структурой файлов.
DATA_DIR = 'C:\\Users\\Sirius\\Desktop\\neuronetwork\\GOTOVO' # Например: 'C:/Users/User/Desktop/my_dataset'
IMAGE_SUBDIR = 'processed_oil_spill_images' # Поддиректория, где хранятся TIF изображения
MASK_SUBDIR = 'processed_mask_oil_spill_images' # Поддиректория, где хранятся пиксельные маски

IMAGE_HEIGHT = 256 # Укажите желаемую высоту изображений после изменения размера
IMAGE_WIDTH = 256 # Укажите желаемую ширину изображений после изменения размера
NUM_CHANNELS = 1 # Теперь у нас обычные TIF изображения.
                 # Если они черно-белые (градации серого), оставьте 1.
                 # Если они цветные (RGB), измените на 3.
NUM_CLASSES = 1 # Для бинарной маски (например, объект/фон) - 1 класс.
                # Если маска имеет несколько классов (например, разные типы объектов),
                # измените на количество классов и используйте 'categorical_crossentropy'
                # в качестве функции потерь.

# --- 2. Загрузка и предварительная обработка данных ---

def load_image_and_mask(image_path, mask_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    """
    Загружает TIF изображение и соответствующую пиксельную маску.
    Эта функция адаптирована для обычных TIF изображений.
    """
    try:
        # Для загрузки изображений будем использовать rasterio для большей надежности с TIFF
        with rasterio.open(image_path) as src:
            image = src.read() # Читаем все каналы

        # rasterio читает в формате (каналы, высота, ширина),
        # а OpenCV и Keras ожидают (высота, ширина, каналы).
        # Если каналов несколько, переставляем оси.
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0)) # Переставляем оси: (C, H, W) -> (H, W, C)
        elif image.ndim == 2:
            image = np.expand_dims(image, axis=-1) # Если монохромное (H, W), делаем (H, W, 1)

        # Проверка и преобразование количества каналов
        current_channels = image.shape[-1]
        if current_channels != NUM_CHANNELS:
            if current_channels == 1 and NUM_CHANNELS == 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif current_channels == 3 and NUM_CHANNELS == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=-1) # Добавляем измерение для канала
            else:
                raise ValueError(f"Несоответствие каналов: Загружено {current_channels}, ожидается {NUM_CHANNELS}")

        # Загрузка маски (обычно одноканальное изображение)
        # Для масок можно по-прежнему использовать cv2.imread, если они простые PNG/TIF
        # или использовать rasterio, если маски также сложны
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
             # Попробуем rasterio для маски, если cv2.imread не сработал
            with rasterio.open(mask_path) as src_mask:
                mask = src_mask.read(1) # Читаем первый канал маски

        # Изменение размера
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST) # Для масок используем INTER_NEAREST, чтобы сохранить дискретные значения

        # Нормализация изображений (0-1)
        # Важно: если исходные данные TIFF имеют другую глубину (например, 16 бит),
        # 255.0 будет неправильной шкалой. Для спутниковых данных часто нужно масштабировать по max_val.
        # Для 8-битных данных 255.0 корректно.
        image = image.astype(np.float32) / 255.0
        
        # Нормализация маски (0 или 1 для бинарной маски)
        mask = mask.astype(np.float32)
        # Убедитесь, что маска содержит только значения 0 и 1 (или 0 и 255, которые затем делятся на 255)
        if np.max(mask) > 1.0: # Если маска в диапазоне 0-255
            mask /= 255.0
        mask = np.expand_dims(mask, axis=-1) # Добавляем измерение для канала маски

        return image, mask
    except Exception as e:
        print(f"Ошибка при загрузке или обработке {image_path} или {mask_path}: {e}")
        return None, None

def load_single_image_for_prediction(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    """
    Загружает одно TIF изображение для предсказания с использованием rasterio.
    """
    try:
        with rasterio.open(image_path) as src:
            image = src.read() # Читаем все каналы

        if image is None: # Проверка на случай, если rasterio вернул пустой объект
            raise FileNotFoundError(f"Изображение не найдено или не может быть прочитано по пути: {image_path}")

        # rasterio читает в формате (каналы, высота, ширина),
        # а OpenCV и Keras ожидают (высота, ширина, каналы).
        # Если каналов несколько, переставляем оси.
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0)) # Переставляем оси: (C, H, W) -> (H, W, C)
        elif image.ndim == 2:
            image = np.expand_dims(image, axis=-1) # Если монохромное (H, W), делаем (H, W, 1)

        # Проверка и преобразование количества каналов для соответствия NUM_CHANNELS
        current_channels = image.shape[-1]
        if current_channels != NUM_CHANNELS:
            if current_channels == 1 and NUM_CHANNELS == 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif current_channels == 3 and NUM_CHANNELS == 1:
                # Если ожидается 1 канал, но загружено 3, преобразуем в градации серого
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=-1) # Добавляем измерение для канала
            else:
                raise ValueError(f"Несоответствие каналов: Загружено {current_channels}, ожидается {NUM_CHANNELS}")
        
        # Изменение размера
        image = cv2.resize(image, target_size)

        # Нормализация изображений (0-1)
        # Если ваши TIFF файлы имеют глубину 16 бит или более,
        # то вместо 255.0 нужно использовать максимальное значение данных (например, 65535.0 для 16 бит).
        # Если данные уже в диапазоне 0-255, то 255.0 подходит.
        image = image.astype(np.float32) / 255.0

        return image
    except Exception as e:
        print(f"Ошибка при загрузке или обработке {image_path}: {e}")
        return None

def remove_geotiff_metadata(input_filepath, output_filepath):
    """
    Удаляет географические метаданные (CRS и Transform) из файла GeoTIFF.

    Args:
        input_filepath (str): Путь к входному файлу GeoTIFF.
        output_filepath (str): Путь для сохранения выходного файла GeoTIFF без метаданных.
    """
    try:
        with rasterio.open(input_filepath, photometric="RG") as src:
            # Получаем профиль (метаданные) исходного GeoTIFF
            profile = src.profile

            # Удаляем географические метаданные: CRS (система координат) и Transform (аффинное преобразование)
            if 'crs' in profile:
                del profile['crs']
            if 'transform' in profile:
                del profile['transform']

            # Открываем новый файл для записи с измененным профилем
            # Важно: 'driver' должен быть указан, например, 'GTiff'
            profile['driver'] = 'GTiff'

            with rasterio.open(output_filepath, 'w', **profile) as dst:
                # Читаем все полосы данных из исходного файла и записываем их в новый
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)
        print(f"Географические метаданные успешно удалены. Файл сохранен как: {output_filepath}")

    except rasterio.errors.RasterioIOError as e:
        print(f"Ошибка ввода/вывода при работе с файлом: {e}")
        print("Пожалуйста, убедитесь, что путь к файлу корректен и файл существует.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")

def load_dataset(data_dir, image_subdir, mask_subdir):
    """
    Собирает пути ко всем изображениям и маскам в датасете.
    """
    image_paths = sorted([os.path.join(data_dir, image_subdir, f) for f in os.listdir(os.path.join(data_dir, image_subdir)) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
    mask_paths = sorted([os.path.join(data_dir, mask_subdir, f) for f in os.listdir(os.path.join(data_dir, mask_subdir)) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])

    # Убедитесь, что количество изображений и масок совпадает
    if len(image_paths) != len(mask_paths):
        raise ValueError("Количество изображений и масок не совпадает!")

    images = []
    masks = []
    for i in range(len(image_paths)):
        img, msk = load_image_and_mask(image_paths[i], mask_paths[i])
        if img is not None and msk is not None:
            images.append(img)
            masks.append(msk)

    return np.array(images), np.array(masks)

print("Загрузка данных...")
try:
    # --- Пример использования remove_geotiff_metadata (закомментировано по умолчанию) ---
    # Если вам нужно удалить метаданные из входных файлов перед обучением или предсказанием:
    # input_geotiff_to_clean = "C:\\Users\\Sirius\\Desktop\\neuronetwork\\GOTOVO2\\S1A_3channel.tif"
    # output_geotiff_cleaned = "C:\\Users\\Sirius\\Desktop\\neuronetwork\\GOTOVO2\\S1A_3channel_cleaned.tif"
    # remove_geotiff_metadata(input_geotiff_to_clean, output_geotiff_cleaned)
    # Теперь вы можете использовать output_geotiff_cleaned для загрузки, например:
    # new_image_path = output_geotiff_cleaned
    # -----------------------------------------------------------------------------------

    X, y = load_dataset(DATA_DIR, IMAGE_SUBDIR, MASK_SUBDIR)
    print(f"Загружено {len(X)} изображений и масок.")
    print(f"Форма изображений: {X.shape}") # Ожидается (количество_образцов, высота, ширина, количество_каналов)
    print(f"Форма масок: {y.shape}") # Ожидается (количество_образцов, высота, ширина, 

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Обучающая выборка: {X_train.shape}, {y_train.shape}")
    print(f"Валидационная выборка: {X_val.shape}, {y_val.shape}")


except Exception as e:
    print(f"Произошла ошибка при загрузке или разделении данных: {e}")
    print("Пожалуйста, проверьте пути к файлам и формат ваших данных.")
    # Выход из программы или использование заглушечных данных для демонстрации модели
    X_train = np.random.rand(10, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
    y_train = np.random.randint(0, 2, size=(10, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES))
    X_val = np.random.rand(2, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
    y_val = np.random.randint(0, 2, size=(2, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES))
    print("Используются заглушечные данные для продолжения демонстрации модели.")


# --- 3. Определение архитектуры сверточной нейросети (U-Net-подобная) ---

def unet_model(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES):
    inputs = keras.Input(input_size)

    # Encoder (Путь сжатия)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs) #1
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1) #2
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2) #3
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3) #4
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck (Дно)
    conv5 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool4) #5
    conv5 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv5)

# Decoder (Путь расширения)
    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.Conv2D(256, 2, activation='relu', padding='same')(up6)
    merge6 = layers.concatenate([conv4, up6], axis=3) # Skip connection
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.Conv2D(128, 2, activation='relu', padding='same')(up7)
    merge7 = layers.concatenate([conv3, up7], axis=3) # Skip connection
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = layers.UpSampling2D(size=(2, 2))(conv7)
    up8 = layers.Conv2D(64, 2, activation='relu', padding='same')(up8)
    merge8 = layers.concatenate([conv2, up8], axis=3) # Skip connection
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = layers.UpSampling2D(size=(2, 2))(conv8)
    up9 = layers.Conv2D(32, 2, activation='relu', padding='same')(up9)
    merge9 = layers.concatenate([conv1, up9], axis=3) # Skip connection
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)

    # Выходной слой
    # Для бинарной сегментации (один класс маски):
    if num_classes == 1:
        outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(conv9)
        loss_function = 'binary_crossentropy'
    # Для многоклассовой сегментации:
    else:
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv9)
        loss_function = 'categorical_crossentropy'

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model, loss_function

print("Создание модели нейросети...")
model, loss_func = unet_model()
model.summary()

# --- 4. Компиляция модели ---
print("Компиляция модели...")
model.compile(optimizer='adam', loss=loss_func, metrics=['accuracy'])


# --- 5. Обучение модели ---
print("Обучение модели...")
# Вы можете настроить количество эпох и размер батча
EPOCHS = 20
BATCH_SIZE = 8

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1
)

# --- 6. Оценка модели (опционально) ---
print("\nОценка модели на валидационных данных:")
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Потери на валидационной выборке: {loss:.4f}")
print(f"Точность на валидационной выборке: {accuracy:.4f}")

# --- 8. Использование модели для предсказаний (пример) ---
# Если у вас есть новые изображения для предсказания:
new_image_path = "C:\\Users\\Sirius\\Desktop\\neuronetwork\\GOTOVO2\\S1A_3channel.tif" # Путь к вашему изображению

# Загружаем изображение для предсказания (нормализованное для модели)
new_image_for_model = load_single_image_for_prediction(new_image_path)

if new_image_for_model is not None:
    # Добавляем измерение для батча
    new_image_for_model = np.expand_dims(new_image_for_model, axis=0)
    
    # Выполняем предсказание
    prediction = model.predict(new_image_for_model)
    
    # Преобразуем предсказание в маску изображения (значения 0 или 255)
    if NUM_CLASSES == 1:
        prediction_mask = (prediction[0] * 255).astype(np.uint8)
    else:
        prediction_mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)
        # Для визуализации разных классов, можно масштабировать значения
        prediction_mask = (prediction_mask * (255 // (NUM_CLASSES - 1))).astype(np.uint8)

    # --- Сохранение бинарной маски ---
    # Убедимся, что prediction_mask имеет 2 измерения (H, W) для сохранения
    # Если она (H, W, 1), убираем последний канал
    if prediction_mask.ndim == 3 and prediction_mask.shape[-1] == 1:
        binary_output_mask = prediction_mask.squeeze(axis=-1)
    else:
        binary_output_mask = prediction_mask

    # Выводим бинарный массив маски в консоль
    print("\nPredicted Binary Mask (Array):\n")
    print(binary_output_mask)
    print("\n")

    # Сохраняем результат в формате TIF
    output_filename = 'predicted_binary_mask.tif'
    cv2.imwrite(output_filename, binary_output_mask)
    print(f"Предсказание выполнено и сохранено как {output_filename} (бинарная маска).")

else:
    print(f"Не удалось загрузить изображение для предсказания: {new_image_path}")