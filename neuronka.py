import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2 # Используем OpenCV для загрузки изображений, если они в форматах, поддерживаемых cv2 (например, .png, .tif)
# from PIL import Image # Альтернатива для PIL, если изображения в других форматах или предпочтительнее

# --- 1. Конфигурация и пути к данным ---
# Вам нужно будет изменить эти пути в соответствии с вашей локальной структурой файлов.
DATA_DIR = 'C:\\Users\\Sirius\\Desktop\\neuronetwork\\01_Train_Val_Oil_Spill_images' 
IMAGE_SUBDIR = 'Oil' # Поддиректория, где хранятся двухканальные изображения
MASK_SUBDIR = 'Mask oil'   # Поддиректория, где хранятся пиксельные маски

IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
NUM_CHANNELS = 2   
NUM_CLASSES = 1    


# --- 2. Загрузка и предварительная обработка данных ---

def load_image_and_mask(image_path, mask_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    """
    Загружает двухканальное изображение и соответствующую пиксельную маску.
    Вам нужно будет адаптировать эту функцию под формат ваших файлов.
    Например, если ваши двухканальные изображения хранятся как отдельные файлы
    для каждого канала или как многоканальные TIFF.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # Загрузка без изменений, сохраняя каналы
        if image.shape[-1] != NUM_CHANNELS:
            print(f"Предупреждение: Изображение {image_path} имеет {image.shape[-1]} каналов, ожидалось {NUM_CHANNELS}.")
        # Загрузка маски 
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Изменение размера
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST) # Для масок используем INTER_NEAREST, чтобы сохранить дискретные значения

        # Нормализация изображений (0-1)
        image = image.astype(np.float32) / 255.0
        # Нормализация маски (0 или 1 для бинарной маски)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1) 

        return image, mask
    except Exception as e:
        print(f"Ошибка при загрузке или обработке {image_path} или {mask_path}: {e}")
        return None, None
def load_dataset(data_dir, image_subdir, mask_subdir):
    """
    Собирает пути ко всем изображениям и маскам в датасете.
    """
    image_paths = sorted([os.path.join(data_dir, image_subdir, f) for f in os.listdir(os.path.join(data_dir, image_subdir)) if f.endswith(('.png', '.tif', '.jpg'))])
    mask_paths = sorted([os.path.join(data_dir, mask_subdir, f) for f in os.listdir(os.path.join(data_dir, mask_subdir)) if f.endswith(('.png', '.tif', '.jpg'))])

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
    X, y = load_dataset(DATA_DIR, IMAGE_SUBDIR, MASK_SUBDIR)
    print(f"Загружено {len(X)} изображений и масок.")
    print(f"Форма изображений: {X.shape}") # Ожидается (количество_образцов, высота, ширина, количество_каналов)
    print(f"Форма масок: {y.shape}")     # Ожидается (количество_образцов, высота, ширина, 1)

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
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck (Дно)
    conv5 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool4)
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
EPOCHS = 30000
BATCH_SIZE = 1000

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1
)


print("\nОценка модели на валидационных данных:")
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Потери на валидационной выборке: {loss:.4f}")
print(f"Точность на валидационной выборке: {accuracy:.4f}")







