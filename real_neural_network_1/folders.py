import os
import shutil
from pathlib import Path

def merge_directories(source_dirs, target_dir):
    """
    Объединяет содержимое нескольких директорий в одну целевую директорию.

    :param source_dirs: список путей к исходным директориям
    :param target_dir: путь к целевой директории
    """
    os.makedirs(target_dir, exist_ok=True)

    for src_dir in source_dirs:
        if not os.path.isdir(src_dir):
            print(f"Пропущено: {src_dir} не является директорией")
            continue

        for root, _, files in os.walk(src_dir):
            rel_path = os.path.relpath(root, src_dir)
            target_subdir = os.path.join(target_dir, rel_path)
            os.makedirs(target_subdir, exist_ok=True)

            for file in files:
                src_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_subdir, file)

                # Предотвращаем перезапись файлов с одинаковыми именами
                if os.path.exists(target_file_path):
                    base, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(target_file_path):
                        target_file_path = os.path.join(
                            target_subdir, f"{base}_{i}{ext}"
                        )
                        i += 1

                shutil.copy2(src_file_path, target_file_path)
                print(f"Копирован: {src_file_path} -> {target_file_path}")

def get_files(input):
    for fd, _, fns in os.walk(input):
       for fn in fns:
            yield os.path.join(fd, fn)

# Пример использования
if __name__ == "__main__":
    # main_dir = Path()
    # source_dir = main_dir / "cropped_images"
    # source_dirs = source_dir / ""  # Замените на ваши директории
    # target_dir = "./merged_dir"
    # merge_directories(source_dirs, target_dir)
    
    for filepath in get_files("C:\\Users\\Sirius\\Desktop\\snaps\\snaps"):
        print(filepath)

