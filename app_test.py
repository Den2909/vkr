import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import numpy as np

class_names = [
    'Вскрышной грунт', 'Глина кирпичная', 'Грунт гранитный скальный ГЛЫБОВЫЙ фр.0-500',
    'Грунт гранитный скальный глыбовый', 'Грунт гранитный скальный гравийный фр.0-300',
    'Дизельное топливо', 'Калиброванный-ДРОБЛЕННЫЙ скальный грунт фр.0-200',
    'Отсев(фракция 0-3)', 'Отсев(фракция 0-5)', 'ПустойКузов', 'Супесь',
    'ЩПС фракция 0-10', 'ЩПС фракция 0-120', 'ЩПС фракция 0-20', 'ЩПС фракция 0-40',
    'ЩПС фракция 0-80', 'Щебень фракция 03-10', 'Щебень фракция 20-40',
    'Щебень фракция 40-70', 'Щебень фракция 5-20', 'Щебень фракция 70-120'
]
num_classes = len(class_names)
 
# Определяем устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Определение модели
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# Модель с пространственным вниманием
class MyNetWithSpatialAttention(nn.Module):
    def __init__(self, in_channels: int = 3, num_of_classes: int = len(class_names)):  # Исправлено на len(class_names)
        super(MyNetWithSpatialAttention, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b1')
        in_features_efficient_net = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Identity()  # Убираем финальный классификатор EfficientNet
        self.spatial_attention = SpatialAttention()
        self.base_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features_efficient_net, 1024),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_of_classes)  # Исправлено на num_of_classes
        )

    def forward(self, x):
        x_efficient_net = self.efficient_net.extract_features(x)
        spatial_attention = self.spatial_attention(x_efficient_net)
        x_efficient_net = x_efficient_net * spatial_attention  # Применяем внимание
        x_efficient_net = x_efficient_net.mean([2, 3])  # Глобальная агрегация
        x_base_model = self.base_classifier(x_efficient_net)
        x = self.classifier(x_base_model)
        return x

# Создаем модель
model = MyNetWithSpatialAttention(num_of_classes=len(class_names)).to(device)

# Функция загрузки модели
def load_model(model, checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=device)  # исправлено
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Модель успешно загружена из {checkpoint_path}")
    return model

# Загружаем веса
try:
    model = load_model(model, 'checkpoint.pth')
except FileNotFoundError:
    print("Ошибка: Файл checkpoint.pth не найден!")

# Предобработка изображений (убраны аугментации)
preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
])

# Функция предсказания
def predict_class(image):
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Интерфейс Streamlit
def main():
    st.title("Классификация фракции щебня")
    uploaded_image = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Загруженное изображение", use_column_width=True)
        pil_image = Image.open(uploaded_image).convert('RGB')
        pil_image = pil_image.copy()

        if st.button("Классифицировать"):
            predicted_class = predict_class(pil_image)
            st.success(f"Класс фракции: {class_names[predicted_class]}")

if __name__ == "__main__":
    main()
