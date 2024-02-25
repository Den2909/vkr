import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pickle
from torch.optim import Optimizer
import torch.nn as nn


class_names = ['Песок из отсевов дробления 0-5 мм', 'ЩПС фр. 0-20 мм НЕГОСТ', 'Щебень 25-60 мм', 'Щебень фр. 20-40 мм', 'Щебень фр. 40-70 мм', 'Щебень фр. 5-20 мм', 'Щебень фракция 8-16']
num_classes = len(class_names)
# Загрузка предварительно обученной модели EfficientNet
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

class MyNetWithSpatialAttention(nn.Module):
    def __init__(self, in_channels: int = 3, num_of_classes: int = 7):
        super(MyNetWithSpatialAttention, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b1')
        in_features_efficient_net = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Identity()
        self.spatial_attention = SpatialAttention()
        self.base_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features_efficient_net, 1024),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_of_classes)
        )

    def forward(self, x):
        x_efficient_net = self.efficient_net.extract_features(x)
        spatial_attention = self.spatial_attention(x_efficient_net)
        x_efficient_net = x_efficient_net * spatial_attention
        x_efficient_net = x_efficient_net.mean([2, 3])
        x_base_model = self.base_classifier(x_efficient_net)
        x = self.classifier(x_base_model)
        return x
    


# class LinearLR:
    def __init__(self, optimizer: Optimizer, start_lr: float, end_lr: float, total_iters: int, last_epoch: int = -1, verbose: bool = False):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_iters = total_iters
        self.last_epoch = last_epoch
        self.verbose = verbose

        if last_epoch == -1:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = start_lr
        else:
            self.step()

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.end_lr + (self.start_lr - self.end_lr) * (1 - self.last_epoch / self.total_iters)
        self.last_epoch += 1

        if self.verbose:
            print(f"Learning Rate: {self.get_last_lr()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# criterion = nn.CrossEntropyLoss()
# initial_lr = 0.002
# final_lr = 0.0001
# num_epochs = 90
# #total_iterations = len(train_loader) * num_epochs*100
# total_iterations = num_epochs

model = MyNetWithSpatialAttention(num_of_classes=len(class_names)).to(device)
# optimizer = optim.Adam(model.parameters(), lr=initial_lr)


# lr_scheduler = LinearLR(optimizer, initial_lr, final_lr, total_iterations)

def load_model(model, checkpoint_path, map_location=None):
    with open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Модель успешно загружена из {checkpoint_path}")
    return model

loaded_model = load_model(model, 'checkpoint.pth')



model.eval()

# Преобразование изображения для входа в модель
preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation((-30, 30)),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomVerticalFlip(p=0.5),
])

def predict_class(image):
    # Применение преобразований
    input_tensor = preprocess(image).unsqueeze(0)

    # Получение предсказания
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    
    return predicted_class.item()

def main():
    st.title("Фракция груза классификации")

    # Загрузка изображения
    uploaded_image = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Отображение загруженного изображения
        st.image(uploaded_image, caption="Загруженное изображение", use_column_width=True)

        # Преобразование в формат PIL
        pil_image = Image.open(uploaded_image).convert('RGB')

        # Кнопка для классификации
        if st.button("Классифицировать"):
            # Предсказание класса
            predicted_class = predict_class(pil_image)

            # Отображение результата
            st.success(f"Класс фракции: {predicted_class}")

if __name__ == "__main__":
    main()