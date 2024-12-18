import torch
from torchvision import transforms, models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Путь к датасету и модели
dataset_path = 'test'  # Путь к изображениям
model_path = 'model.pth'  # Путь к сохранённой модели

# Трансформация для тестирования (ЧБ и нормализация)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Загрузка обученной модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Для ЧБ изображений
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 13)  # 13 классов карт
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Названия классов (адаптируй под свои классы)
class_names = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q']


# Функция для предсказания и отображения всех изображений
def predict_and_display_all_images(image_folder, model, transform, class_names, device):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                   img.endswith(('png', 'jpg', 'jpeg'))]

    num_images = len(image_paths)
    cols = 5  # Количество столбцов
    rows = (num_images + cols - 1) // cols  # Вычисляем количество строк

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()  # Превращаем оси в одномерный массив для удобства

    for i, img_path in enumerate(image_paths):
        # Загружаем изображение и преобразуем
        image = Image.open(img_path).convert('L')
        input_image = transform(image).unsqueeze(0).to(device)

        # Предсказание
        with torch.no_grad():
            output = model(input_image)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]

        # Отображаем изображение и результат
        axes[i].imshow(np.array(image), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Predicted: {predicted_class}")

    # Удаляем лишние пустые графики
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Запуск функции
test_image_folder = os.path.join(dataset_path, '')  # Укажи папку с изображениями
predict_and_display_all_images(test_image_folder, model, transform, class_names, device)
