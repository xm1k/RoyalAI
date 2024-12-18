import torch
from torchvision import transforms, models
from PIL import Image
import os

model_path = 'suit_detector/model.pth'
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 5)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

class_names = ['clubs', 'diamonds', 'hearts', 'not-card', 'spades']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class

def suit_detect(input_folder='detected_cards'):
    class_vectors = []
    image_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda f: int(f.split('_')[-1].split('.')[0])
    )
    for image_name in image_files:
        image_path = os.path.join(input_folder, image_name)
        predicted_class = predict_image(image_path)
        class_vectors.append(predicted_class)
    return class_vectors