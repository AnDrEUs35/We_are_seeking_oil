import os
import shutil
import sys
from pathlib import Path

# ===== КОНФИГУРАЦИЯ =====
INPUT_DIR = Path.cwd() / sys.argv[1]  # Входная директория с бинарными масками
OUTPUT_DIR = Path.cwd() / (sys.argv[1] + "_png_Oil") 
PREFIX = "mask_"  # Ваш префикс
# ========================

# Проверка существования директорий
if not os.path.exists(INPUT_DIR):
    print(f" Ошибка: Входная директория не существует: {INPUT_DIR}")
    exit(1)

if not os.path.exists(OUTPUT_DIR):
    print(f" Выходная директория не существует. Создаю: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR)

# Получение списка файлов во входной директории
files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

print(f"Обработка {len(files)} файлов из {INPUT_DIR}...")
print(f"Результат будет сохранен в {OUTPUT_DIR} с префиксом '{PREFIX}'")

success = 0
skipped = 0
errors = 0

for filename in files:
    try:
        # Формирование путей
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, PREFIX + filename)
        
        # Проверка существования файла в выходной директории
        if os.path.exists(output_path):
            print(f"⚠️ Пропуск: {filename} (уже существует в выходной директории)")
            skipped += 1
            continue
            
        # Копирование файла с новым именем
        shutil.copy2(input_path, output_path)
        print(f" {filename} → {os.path.basename(output_path)}")
        success += 1
        
    except Exception as e:
        print(f" Ошибка с {filename}: {str(e)}")
        errors += 1

# Итоговый отчет
print("\n" + "="*50)
print(f"Итоги обработки:")
print(f"• Успешно скопировано: {success} файлов")
print(f"• Пропущено: {skipped} файлов (конфликты имен)")
print(f"• Ошибок: {errors}")
print(f"• Исходные файлы остались в: {INPUT_DIR}")
print(f"• Результаты сохранены в: {OUTPUT_DIR}")
print("="*50)