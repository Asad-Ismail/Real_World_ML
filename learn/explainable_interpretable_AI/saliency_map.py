import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load an image and preprocess it
img = Image.open("image.jpg")
img_t = transform(img)
img_t = img_t.unsqueeze(0)

# Enable gradients computation
img_t.requires_grad_(True)

# Forward pass
out = model(img_t)

# Get the index of the max log-probability
score, index = torch.max(out, 1)

# Backward pass for the index of the max log-probability
score.backward()
saliency, _ = torch.max(img_t.grad.data.abs(), dim=1)

# Plot the saliency map
plt.imshow(saliency[0], cmap=plt.cm.hot)
plt.axis('off')
plt.show()
