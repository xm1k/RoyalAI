import torch
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt

# Загрузка модели
model_path = 'model.pth'
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 5)  # Количество классов: 5
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Названия классов
class_names = ['clubs', 'diamonds', 'hearts', 'not-card', 'spades']

# Трансформации для подготовки изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Функция для предсказания класса
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Добавляем batch dimension
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    return img, predicted_class

# Функция для отображения изображений с подписями
def show_images_with_predictions(input_folder='test'):
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    plt.figure(figsize=(15, 15))
    for i, image_name in enumerate(image_files):
        image_path = os.path.join(input_folder, image_name)
        print(f"Обрабатываю: {image_name}")

        # Предсказание класса
        img, predicted_class = predict_image(image_path)
        print(f"Класс для {image_name}: {predicted_class}")

        # Отображаем изображение
        plt.subplot(7, 6, i + 1)  # Задаем сетку 4x4 (можно изменить под количество изображений)
        plt.imshow(img)
        plt.title(predicted_class, color='red')
        plt.axis('off')  # Убираем оси

    plt.tight_layout()
    plt.show()

# Отображение изображений
show_images_with_predictions(input_folder='test')
