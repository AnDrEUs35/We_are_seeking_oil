import os
import shutil
import random

# Пути к исходным данным
images_dir = 'dataset/cropped_images_png'
masks_dir = 'dataset/cropped_masks_png'

# Пути для новой выборки
output_images_dir = 'ready/im_val'
output_masks_dir = 'ready/mask_val'

# Создаем папки, если их нет
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# Получаем список всех изображений
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Выбираем 25% случайным образом
num_test = int(len(image_files) * 0.25)
test_files = random.sample(image_files, num_test)

for file in test_files:
    # Пути к изображениям и маскам
    src_image_path = os.path.join(images_dir, file)
    src_mask_path = os.path.join(masks_dir, file)

    dst_image_path = os.path.join(output_images_dir, file)
    dst_mask_path = os.path.join(output_masks_dir, file)

    # Переносим файлы
    if os.path.exists(src_mask_path):
        shutil.move(src_image_path, dst_image_path)
        shutil.move(src_mask_path, dst_mask_path)
    else:
        print(f"Маска для изображения {file} не найдена, пропуск.")

print("Разделение завершено.")
