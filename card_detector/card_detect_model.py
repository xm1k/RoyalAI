import torch
from PIL import Image
import matplotlib.pyplot as plt
import os

model_path = 'model.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.conf = 0.25
model.iou = 0.3

def process_all_images(input_folder='poker', output_folder='detected_cards'):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpeg', '.jpg'))]
    for image_name in image_files:
        image_path = os.path.join(input_folder, image_name)
        detect_and_extract_cards(image_path, output_folder)

def detect_and_extract_cards(image_path, output_dir='detected_cards'):
    img = Image.open(image_path).convert('RGB')
    results = model(img)
    predictions = results.xyxy[0]
    if len(predictions) > 0:
        for idx, pred in enumerate(predictions):
            x_min, y_min, x_max, y_max, conf, cls = pred.tolist()
            cropped_card = img.crop((x_min, y_min, x_max, y_max))
            card_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_card_{idx + 1}.png"
            card_path = os.path.join(output_dir, card_filename)
            cropped_card.save(card_path)
            plt.figure()
            plt.title(f"Card {idx + 1}, Confidence: {conf:.2f}")
            plt.imshow(cropped_card)
            plt.axis('off')
            plt.show()
    else:
        print(f"Карты не найдены на изображении: {image_path}")
    results.render()
    plt.imshow(Image.fromarray(results.ims[0]))
    plt.axis('off')
    plt.show()

process_all_images(input_folder='poker', output_folder='detected_cards')
