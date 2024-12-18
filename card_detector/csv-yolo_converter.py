import pandas as pd
import os
import json
from PIL import Image


def convert_csv_to_yolo(csv_path, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем CSV-файл
    annotations = pd.read_csv(csv_path)

    grouped_annotations = annotations.groupby('filename')  # Группируем по файлам

    for image_name, group in grouped_annotations:
        # Путь к изображению
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Изображение отсутствует: {image_path}")
            continue

        # Получаем размеры изображения
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Список аннотаций для текущего изображения
        yolo_annotations = []

        for _, row in group.iterrows():
            shape_attributes = row['region_shape_attributes']

            # Преобразуем строку JSON в словарь
            try:
                shape_data = json.loads(shape_attributes)
            except json.JSONDecodeError:
                print(f"Ошибка декодирования JSON в строке: {shape_attributes}")
                continue

            # Проверяем, что это прямоугольник
            if shape_data.get('name') != 'rect':
                print(f"Пропущена фигура: {shape_data.get('name')}")
                continue

            # Извлекаем координаты прямоугольника
            if not all(key in shape_data for key in ['x', 'y', 'width', 'height']):
                print(f"Недостаточно данных в аннотации: {shape_data}")
                continue

            x = shape_data['x']
            y = shape_data['y']
            width = shape_data['width']
            height = shape_data['height']

            # Нормализуем координаты
            x_center = (x + width / 2) / img_width
            y_center = (y + height / 2) / img_height
            norm_width = width / img_width
            norm_height = height / img_height

            # Формат аннотации для YOLO: class_id x_center y_center width height
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

        # Если есть аннотации, сохраняем их
        if yolo_annotations:
            txt_filename = os.path.splitext(image_name)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_annotations))
        else:
            # Если аннотаций нет, создаем пустой файл
            txt_filename = os.path.splitext(image_name)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)
            with open(txt_path, 'w') as f:
                # Пустой файл, так как объекта нет
                f.write("")


# Пример использования
convert_csv_to_yolo(
    csv_path='annotation.csv',
    image_dir='dataset/images',
    output_dir='dataset/labels'
)
