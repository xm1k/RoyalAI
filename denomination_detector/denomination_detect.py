import torch
from torchvision import transforms, models
import torch.nn as nn
import os
from PIL import Image

dataset_path = 'detected_cards'  # Путь к изображениям
model_path = 'denomination_detector/model.pth'  # Путь к сохранённой модели

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 13)  # 13 классов карт
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

class_names = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q']


def den_detect(image_folder = dataset_path, model = model, transform = transform, class_names = class_names, device = device):
    image_paths = sorted(
        [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('png', 'jpg', 'jpeg'))],
        key=lambda img: int(img.split('_')[-1].split('.')[0])
    )
    class_vector = []

    for img_path in image_paths:
        image = Image.open(img_path).convert('L')
        input_image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_image)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]

        class_vector.append(predicted_class)

    return class_vector
