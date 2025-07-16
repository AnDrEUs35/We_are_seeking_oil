#C:\\Users\\Sirius\\Desktop\\neuronetwork\\real_neural_network\\normalno_mask
#C:\\Users\\Sirius\\Desktop\\neuronetwork\\real_neural_network\\Mask_oil


import numpy as np
from pathlib import Path
import rasterio
from PIL import Image
import sys

# ===== КОНФИГУРАЦИЯ =====
INPUT_DIR = Path.cwd() / sys.argv[1]  # Входная директория с бинарными масками
OUTPUT_DIR = Path.cwd() / (sys.argv[1]+"_png")   # Выходная директория для PNG
# ========================

def process_binary_mask(input_path, output_path):
    """Обрабатывает бинарную маску и сохраняет как PNG"""
    try:
        with rasterio.open(input_path) as src:
            # Проверяем количество каналов
            if src.count != 1:
                print(f"  Ожидался 1 канал, найдено {src.count}")
                return False
            
            # Читаем данные
            data = src.read(1)
            
            # Проверяем тип данных
            if data.dtype != np.uint8:
                print(f"  Предупреждение: тип данных {data.dtype}, преобразование в uint8")
                data = data.astype(np.uint8)
            
            # Нормализуем в бинарный формат (0 и 255)
            # Определяем уникальные значения для диагностики
            unique_vals = np.unique(data)
            
            # Если только два значения, считаем что это бинарная маска
            if len(unique_vals) == 2:
                # Находим min и max значения
                min_val = np.min(data)
                max_val = np.max(data)
                # Нормализуем: min->0, max->255
                data = np.where(data == max_val, 255, 0).astype(np.uint8)
            # Если одно значение, считаем фоном (0)
            elif len(unique_vals) == 1:
                data = np.where(data == unique_vals[0], 0, 0).astype(np.uint8)
            # Если больше двух значений, используем порог
            else:
                print("  Более 2 уникальных значений, применение порога 0.5")
                # Бинаризация по порогу (среднее между min и max)
                threshold = (np.min(data) + np.max(data)) / 2
                data = np.where(data > threshold, 255, 0).astype(np.uint8)
            
            # Создаем изображение
            img = Image.fromarray(data, mode='L')  # 'L' - 8-bit grayscale
            img.save(output_path)
            return True
            
    except Exception as e:
        print(f"  Ошибка: {str(e)}")
        return False

def main():
    print("Конвертер бинарных масок GeoTIFF в PNG")
    print(f"Входная директория: {INPUT_DIR}")
    print(f"Выходная директория: {OUTPUT_DIR}\n")
    
    # Проверка директорий
    if not INPUT_DIR.exists():
        print("Ошибка: Входная директория не существует")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Поиск GeoTIFF файлов
    extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    tiff_files = []
    for ext in extensions:
        tiff_files.extend(INPUT_DIR.glob(f'*{ext}'))
    
    if not tiff_files:
        print("Не найдено файлов GeoTIFF")
        return
    
    print(f"Найдено {len(tiff_files)} файлов для обработки\n")
    
    success_count = 0
    error_count = 0
    
    for tiff_path in tiff_files:
        png_path = OUTPUT_DIR / (tiff_path.stem + '.png')
        print(f"Обработка: {tiff_path.name}...", end=' ')
        
        if process_binary_mask(tiff_path, png_path):
            print("Успешно")
            success_count += 1
        else:
            print("Ошибка")
            error_count += 1
    
    print("\nРезультат:")
    print(f"Успешно: {success_count}")
    print(f"Ошибок: {error_count}")
    print(f"Выходная директория: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()