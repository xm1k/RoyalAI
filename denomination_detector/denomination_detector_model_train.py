import torch
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random

# Путь к датасету
dataset_path = 'dataset'

# Фиксируем сиды для воспроизводимости
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Трансформации для обучения (аугментации + ЧБ)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Преобразуем в ЧБ
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=10, translate=(0, 0)),  # Повороты и сдвиги
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Среднее и std для ЧБ
])

# Трансформации для валидации и тестирования
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Загружаем датасет
dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
# Рассчитываем веса классов перед разбиением
class_counts = np.bincount([label for _, label in dataset.samples])
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

# Функция для создания WeightedRandomSampler для подмножества
def create_sampler(subset):
    subset_targets = [dataset.samples[i][1] for i in subset.indices]
    subset_class_counts = np.bincount(subset_targets, minlength=len(class_counts))
    subset_weights = 1. / torch.tensor(subset_class_counts, dtype=torch.float)
    sample_weights = [subset_weights[label] for label in subset_targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Разделение на train, val и test
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Создаём WeightedRandomSampler только для train_loader
train_sampler = create_sampler(train_dataset)

# DataLoader для каждой части
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Загружаем ResNet50 и адаптируем к ЧБ
model = models.resnet50(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Изменяем первый слой для ЧБ

# Заменяем последний слой для 13 классов
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 13)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# Переменная для отслеживания наилучшей точности на валидации
best_val_accuracy = 0.0

# Обучение модели
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    val_loss, val_correct, val_total = 0.0, 0, 0

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    scheduler.step(val_loss)

    print(f"Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%")

    # Сохраняем модель, если точность на валидации улучшилась
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'model.pth')
        print(f"Saved new best model with Val Accuracy: {best_val_accuracy:.2f}%")

# Тестирование модели
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Сохранение финальной модели (если необходимо)
torch.save(model.state_dict(), 'final_model.pth')
