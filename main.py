import urllib.request
import urllib.error # Для обработки ошибок HTTP и URL
import zipfile
import os
import shutil # Для удаления директории, если нужно
import sys # Для sys.stdout.flush()
import time # Для задержки между попытками

# --- Конфигурация ---
# Важно: Замените 'https://example.com/path/to/your_large_dataset.zip'
# на фактическую прямую ссылку на ваш датасет.
# Убедитесь, что это прямая ссылка на файл, а не на веб-страницу.
# Для очень больших файлов (40 ГБ) рекомендуется использовать сжатый формат (например, .zip, .tar.gz).
dataset_url = "https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_images.7z?download=1"
output_filename = "01_Train_Val_Oil_Spill_images.7z" # Имя, под которым файл будет сохранен локально

# Директория для распакованных данных.
# Вы можете изменить это на любой путь на вашем компьютере, например, 'data/'
# или 'C:/Users/YourUser/Documents/MyDataset/'
unzip_directory = "C:\\Users\\Sirius\\Downloads"

# Параметры повторных попыток загрузки
max_retries = 5 # Максимальное количество попыток загрузки
retry_delay_seconds = 10 # Задержка между попытками в секундах

# --- Загрузка файла ---
print(f"Попытка загрузки датасета с: {dataset_url}")
print(f"Файл будет сохранен как: {output_filename}")

download_successful = False
for attempt in range(1, max_retries + 1):
    print(f"\nПопытка загрузки {attempt} из {max_retries}...")
    try:
        # Открываем URL для чтения
        with urllib.request.urlopen(dataset_url) as response:
            # Проверяем, что запрос был успешным (код 200)
            if response.getcode() != 200:
                raise urllib.error.HTTPError(dataset_url, response.getcode(),
                                             response.reason, response.headers, None)

            # Получаем общий размер файла, если он доступен
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded_size = 0
            block_size = 8192 # Размер блока для чтения (8 КБ)

            # Открываем локальный файл для записи в бинарном режиме
            with open(output_filename, 'wb') as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break # Если буфер пуст, значит, файл прочитан до конца
                    out_file.write(buffer)
                    downloaded_size += len(buffer)

                    # Выводим прогресс загрузки
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        # Используем \r для перезаписи строки и end='' для предотвращения новой строки
                        sys.stdout.write(f"\rЗагружено: {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({progress:.2f}%)")
                        sys.stdout.flush() # Принудительно выводим буфер stdout
                    else:
                        # Если размер файла неизвестен, просто показываем загруженный объем
                        sys.stdout.write(f"\rЗагружено: {downloaded_size / (1024*1024):.2f} MB")
                        sys.stdout.flush()

        print(f"\nЗагрузка '{output_filename}' завершена.")
        download_successful = True
        break # Выходим из цикла повторных попыток, если загрузка успешна

    except urllib.error.HTTPError as e:
        print(f"\nОшибка HTTP при загрузке файла (код: {e.code}, причина: {e.reason}): {e}")
    except urllib.error.URLError as e:
        print(f"\nОшибка URL при загрузке файла (причина: {e.reason}): {e}")
    except Exception as e:
        print(f"\nПроизошла непредвиденная ошибка при загрузке файла: {e}")

    # Если загрузка не удалась, удаляем частично загруженный файл перед следующей попыткой
    if os.path.exists(output_filename):
        os.remove(output_filename)
        print(f"Удален частично загруженный файл '{output_filename}'.")
    if attempt < max_retries:
        print(f"Повторная попытка через {retry_delay_seconds} секунд...")
        time.sleep(retry_delay_seconds)
    else:
        print(f"Все {max_retries} попыток загрузки не удались. Выход.")
        sys.exit(1) # Выходим с кодом ошибки

if not download_successful:
    sys.exit(1) # Выходим, если загрузка не была успешной после всех попыток

# --- Проверка размера загруженного файла ---
if os.path.exists(output_filename):
    file_size_bytes = os.path.getsize(output_filename)
    print(f"Размер загруженного файла '{output_filename}': {file_size_bytes / (1024*1024*1024):.2f} GB")
else:
    print(f"Ошибка: Файл '{output_filename}' не был загружен.")
    sys.exit(1)

# --- Распаковка архива ---
# Сначала создадим директорию для распакованных данных, если ее нет.
os.makedirs(unzip_directory, exist_ok=True)

print(f"\nНачинаем распаковку '{output_filename}' в '{unzip_directory}' (это может занять много времени)...")

try:
    # Проверяем, является ли файл ZIP-архивом
    if zipfile.is_zipfile(output_filename):
        with zipfile.ZipFile(output_filename, 'r') as zip_ref:
            # Распаковываем все содержимое архива в указанную директорию
            zip_ref.extractall(unzip_directory)
        print(f"Распаковка '{output_filename}' завершена.")
    else:
        print(f"Ошибка: '{output_filename}' не является действительным ZIP-архивом. Проверьте формат файла.")
        # Если это tar.gz, используйте:
        # import tarfile
        # with tarfile.open(output_filename, "r:gz") as tar_ref:
        #     tar_ref.extractall(unzip_directory)

except zipfile.BadZipFile:
    print(f"Ошибка: '{output_filename}' поврежден или не является действительным ZIP-архивом.")
except Exception as e:
    print(f"Произошла ошибка при распаковке: {e}")
    sys.exit(1)

print(f"\nСодержимое распакованной папки '{unzip_directory}':")
# Выводим список файлов в распакованной директории
for item in os.listdir(unzip_directory):
    print(f"- {item}")

# --- Теперь вы можете загрузить свои данные ---
# import pandas as pd
# try:
#     # Замените 'your_data_file.csv' на фактическое имя файла внутри архива.
#     data_file_path = os.path.join(unzip_directory, 'your_data_file.csv')
#     df = pd.read_csv(data_file_path)
#     print("\nПервые 5 строк загруженного датасета:")
#     print(df.head())
# except FileNotFoundError:
#     print(f"Ошибка: Файл 'your_data_file.csv' не найден в {unzip_directory}. Проверьте имя файла.")
# except Exception as e:
#     print(f"Произошла ошибка при чтении файла: {e}")

# Примечание: Для 40 ГБ данных распаковка и последующая загрузка в память
# могут потребовать много оперативной памяти и места на диске вашего ПК.
# Убедитесь, что у вас достаточно ресурсов.
