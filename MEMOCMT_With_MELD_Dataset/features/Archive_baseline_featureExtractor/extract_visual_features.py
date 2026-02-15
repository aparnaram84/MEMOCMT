"""
Visual feature extraction using ResNet
"""
import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def extract_visual_features(image_path):
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return model(img)
