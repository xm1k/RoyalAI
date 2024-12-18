import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil

model_path = 'card_detector/model.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.conf = 0.25
model.iou = 0.4


def clear_and_extract_cards(input_folder='poker', output_folder='detected_cards'):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Очищаем папку перед новым сохранением
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpeg', '.jpg'))]
    all_coordinates = []
    for image_name in image_files:
        image_path = os.path.join(input_folder, image_name)
        coords = detect_and_extract_cards(image_path, output_folder)
        all_coordinates.extend(coords)

    return all_coordinates


def iou(box1, box2):
    # Compute intersection over union of two bounding boxes
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


def detect_and_extract_cards(image_path, output_dir='detected_cards'):
    img = Image.open(image_path).convert('RGB')
    results = model(img)
    predictions = results.xyxy[0]

    coordinates = []
    if len(predictions) > 0:
        sorted_predictions = sorted(predictions.tolist(), key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

        for idx, pred in enumerate(sorted_predictions):
            x_min, y_min, x_max, y_max, conf, cls = pred
            current_box = (x_min, y_min, x_max, y_max)
            current_area = (x_max - x_min) * (y_max - y_min)

            total_overlap_area = sum(
                iou(current_box, other) * ((other[2] - other[0]) * (other[3] - other[1]))
                for other in coordinates
            )

            new_info = current_area - total_overlap_area
            new_info_percentage = new_info / current_area * 100

            if new_info_percentage > 65:
                coordinates.append(current_box)

                cropped_card = img.crop(current_box)
                card_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_card_{idx + 1}.png"
                card_path = os.path.join(output_dir, card_filename)
                os.makedirs(output_dir, exist_ok=True)
                cropped_card.save(card_path)

    results.render()
    plt.imshow(Image.fromarray(results.ims[0]))
    plt.axis('off')
    plt.show()

    return coordinates



