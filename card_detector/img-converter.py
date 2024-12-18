import os
from PIL import Image

# Пути
input_folder = 'poker'
output_folder = 'dataset/images'

# Создаем выходную папку, если она не существует
os.makedirs(output_folder, exist_ok=True)

# Обрабатываем все изображения в папке 'poker'
for filename in os.listdir(input_folder):
    # Пропускаем файлы, которые не являются изображениями
    if not (filename.endswith('.jpeg') or filename.endswith('.png')):
        continue

    # Путь к изображению
    image_path = os.path.join(input_folder, filename)

    try:
        # Открываем изображение
        img = Image.open(image_path)

        # Получаем минимальную сторону изображения для создания квадрата
        min_side = min(img.size)

        # Создаем квадратное изображение (обрезаем по центру)
        left = (img.width - min_side) / 2
        top = (img.height - min_side) / 2
        right = (img.width + min_side) / 2
        bottom = (img.height + min_side) / 2
        img_cropped = img.crop((left, top, right, bottom))

        # Масштабируем изображение до 640x640
        img_resized = img_cropped.resize((640, 640))

        # Сохраняем обработанное изображение в папку 'dataset/images/train'
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)

        print(f"Обработано: {filename}")

    except Exception as e:
        print(f"Ошибка обработки {filename}: {e}")
