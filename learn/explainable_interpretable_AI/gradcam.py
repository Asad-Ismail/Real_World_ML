import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load a pre-trained model
model = models.resnet18(pretrained=True)
model.eval()  # Set model to evaluation mode

# Preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

# Grad-CAM
def grad_cam(model, img, feature_layer):
    img.requires_grad = True
    feature_maps = {}  # To store forward pass features
    grads = {}  # To store the gradients

    def forward_hook(module, input, output):
        feature_maps["last"] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        grads["last"] = grad_out[0].detach()

    # Attach hooks
    hook_f = feature_layer.register_forward_hook(forward_hook)
    hook_b = feature_layer.register_backward_hook(backward_hook)

    # Forward pass
    outputs = model(img)
    _, pred = outputs.max(dim=1)

    # Zero gradients, backward pass
    model.zero_grad()
    pred.backward()

    # Remove hooks
    hook_f.remove()
    hook_b.remove()

    # Get feature maps and gradients
    gradients = grads["last"]  # Shape: [N, C, H, W]
    activations = feature_maps["last"]  # Shape: [N, C, H, W]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # Global Average Pooling on gradients

    # Weight the activations with gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()  # Average across channels
    heatmap = np.maximum(heatmap.cpu(), 0)  # ReLU on heatmap
    heatmap /= torch.max(heatmap)  # Normalize

    return heatmap.numpy()

img_path = 'image.jpg'  
img = preprocess_image(img_path)
feature_layer = model.layer4[-1]  # Last layer
heatmap = grad_cam(model, img, feature_layer)

# Visualization
plt.matshow(heatmap)
plt.show()

# To overlay the heatmap on the original image, resize the heatmap to the original image size and overlaying it with transparency.
