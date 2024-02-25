import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class_names = ['Песок из отсевов дробления 0-5 мм', 'ЩПС фр. 0-20 мм НЕГОСТ', 'Щебень 25-60 мм', 'Щебень фр. 20-40 мм', 'Щебень фр. 40-70 мм', 'Щебень фр. 5-20 мм', 'Щебень фракция 8-16']
num_classes = len(class_names)

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
    def __init__(self, in_channels: int = 3, num_of_classes: int = num_classes):
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyNetWithSpatialAttention().to(device)

def load_model(model, checkpoint_path, map_location=None):
    with open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Модель успешно загружена из {checkpoint_path}")
    return model

loaded_model = load_model(model, 'checkpoint.pth')

model.eval()

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
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

def main():
    st.title("Классификация фракции щебня")
    uploaded_image = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Загруженное изображение", use_column_width=True)
        pil_image = Image.open(uploaded_image).convert('RGB')

        if st.button("Классифицировать"):
            predicted_class = predict_class(pil_image)
            st.success(f"Класс фракции: {class_names[predicted_class]}")

if __name__ == "__main__":
    main()