import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import models
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Фиксация seed для воспроизводимости
seed = 42
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(seed)

# Путь к вашему датасету
dataset_path = 'dataset'

# Трансформации для подготовки данных
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загружаем весь датасет, используя ImageFolder
dataset = datasets.ImageFolder(root=dataset_path, transform=val_test_transform)

# Разделение на train, val и test
train_idx, temp_idx = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    stratify=dataset.targets,
    random_state=seed
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=np.array(dataset.targets)[temp_idx],
    random_state=seed
)

train_dataset = Subset(dataset, train_idx)
train_dataset.dataset.transform = train_transform  # Применяем аугментации только к train

val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

# Создаем DataLoader для каждой части
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Загружаем предварительно обученную модель ResNet50
model = models.resnet50(pretrained=True)

# Заменяем последний слой для 5 классов
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)

# Переносим модель на GPU, если доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Определяем функцию потерь с учетом баланса классов
class_weights = torch.tensor(
    np.bincount(dataset.targets) / len(dataset.targets), dtype=torch.float
).to(device)
criterion = nn.CrossEntropyLoss(weight=1 / class_weights)

# Оптимизатор и Scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Функция обучения
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy

# Функция валидации/тестирования
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy

# Обучение модели
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion)
    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Сохраняем модель
torch.save(model.state_dict(), 'model.pth')

# Тестирование модели
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
