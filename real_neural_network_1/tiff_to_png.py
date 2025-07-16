#C:\\Users\\Sirius\\Desktop\\neuronetwork\\real_neural_network\\normalno
#C:\\Users\\Sirius\\Desktop\\neuronetwork\\real_neural_network\\Oil


import numpy as np
from pathlib import Path
import rasterio
from PIL import Image
import sys 

# ===== КОНФИГУРАЦИЯ =====
INPUT_DIR = Path.cwd() / sys.argv[1]  # Входная директория с бинарными масками
OUTPUT_DIR = Path.cwd() / (sys.argv[1]+"_png")   # Выходная директория для PNG
# ========================

def normalize_to_8bit(arr):
    """Нормализует данные в 8-битный диапазон (0-255)"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=np.nanmax(arr), neginf=np.nanmin(arr))
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if min_val == max_val:
        return np.zeros_like(arr, dtype=np.uint8)
    
    normalized = (arr - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)

def process_geotiff(input_path, output_path):
    """Обрабатывает GeoTIFF и сохраняет как PNG"""
    try:
        with rasterio.open(input_path) as src:
            # Проверяем количество каналов
            if src.count < 2:
                print(f"  Требуется минимум 2 канала, найдено {src.count}")
                return False
            
            # Читаем первые два канала
            ch1 = src.read(1)
            ch2 = src.read(2)
            
            # Нормализуем в 8-битный диапазон
            ch1_norm = normalize_to_8bit(ch1)
            ch2_norm = normalize_to_8bit(ch2)
            
            # Создаем третий канал как среднее
            ch3_norm = ((ch1_norm.astype(np.uint16) + ch2_norm.astype(np.uint16)) // 2).astype(np.uint8)
            
            # Собираем RGB изображение (H x W x 3)
            rgb = np.dstack((ch1_norm, ch2_norm, ch3_norm))
            
            # Создаем и сохраняем PNG
            img = Image.fromarray(rgb, mode='RGB')
            img.save(output_path)
            return True
            
    except Exception as e:
        print(f"  Ошибка: {str(e)}")
        return False

def main():
    print("Конвертер GeoTIFF в PNG")
    print(f"Вход: {INPUT_DIR}")
    print(f"Выход: {OUTPUT_DIR}\n")
    
    if not INPUT_DIR.exists():
        print("Ошибка: Входная директория не существует")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Находим все GeoTIFF файлы
    tiff_files = []
    for ext in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
        tiff_files.extend(INPUT_DIR.glob(ext))
    
    if not tiff_files:
        print("Не найдено GeoTIFF файлов")
        return
    
    success = 0
    failed = 0
    
    for tiff_path in tiff_files:
        png_path = OUTPUT_DIR / (tiff_path.stem + '.png')
        print(f"Обработка: {tiff_path.name}...", end=' ')
        
        if process_geotiff(tiff_path, png_path):
            print("Успешно")
            success += 1
        else:
            print("Ошибка")
            failed += 1
    
    print("\nРезультат:")
    print(f"Успешно: {success}")
    print(f"Ошибок: {failed}")

if __name__ == "__main__":
    main()