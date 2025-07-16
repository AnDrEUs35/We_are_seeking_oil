from PIL import Image
import numpy as np
import os

def convert_tiff_to_png(tiff_path, png_path):
    """
    Преобразует 32-битный одноканальный TIFF-файл в PNG-файл с 8 битами на канал.

    :param tiff_path: путь к входному TIFF-файлу
    :param png_path: путь к выходному PNG-файлу

    Как работает преобразование:
    1. Открывается TIFF-файл с помощью PIL.Image.open().
    2. Преобразуется в массив numpy с типом float32, чтобы получить доступ к значениям пикселей.
    3. Выполняется min-max нормализация значений пикселей в диапазон [0, 255],
       так как PNG с 8 битами на канал поддерживает максимум 255.
    4. Массив преобразуется в uint8 и сохраняется как PNG-файл в режиме 'L' (grayscale 8-бит).
    """
    with Image.open(tiff_path) as img:
        arr = np.array(img).astype(np.float32)

        min_val, max_val = arr.min(), arr.max()
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val) * 255
        else:
            arr.fill(0)  # если изображение имеет одинаковые значения пикселей

        arr = arr.astype(np.uint8)
        img_out = Image.fromarray(arr, mode='L')
        img_out.save(png_path, format='PNG')

convert_tiff_to_png('path/to/input.tif', 'path/to/output.png')