import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def train_nerf(nerf_model, coords, colors, optimizer, epochs=1000):
    nerf_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_colors = nerf_model(coords)
        loss = torch.mean((pred_colors - colors) ** 2)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

def generate_image(nerf_model, img_width, img_height, focal_length, eye, look_at, up):
    nerf_model.eval()
    aspect_ratio = float(img_width) / img_height
    camera_direction = (look_at - eye).float()
    camera_direction = camera_direction / torch.norm(camera_direction)
    camera_right = torch.linalg.cross(camera_direction, up.float())
    camera_right = camera_right / torch.norm(camera_right)
    camera_up = torch.linalg.cross(camera_right, camera_direction)

    x = torch.linspace(-aspect_ratio, aspect_ratio, img_width)
    y = torch.linspace(-1.0, 1.0, img_height)
    yv, xv = torch.meshgrid(y, x, indexing="ij")

    coords = torch.stack(
        [
            xv.reshape(-1),
            yv.reshape(-1),
            torch.full((img_height * img_width,), focal_length),
        ],
        dim=1,
    )

    camera_basis = torch.stack([camera_right, camera_up, camera_direction], dim=1)
    world_coords = coords @ camera_basis.T + eye.float()

    colors = nerf_model(world_coords)
    colors = colors.view(img_height, img_width, 3).detach().numpy()

    return np.clip(colors, 0, 1)

def main():
    ensure_results_dir()
    input_dim = 3
    hidden_dim = 256
    output_dim = 3
    num_layers = 4

    nerf_model = MLP(input_dim, hidden_dim, output_dim, num_layers)
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=1e-3)

    # Prepare your training data: coordinates (3D points) and colors (RGB)
    coords = torch.randn(1000, 3)
    colors = torch.randn(1000, 3)

    train_nerf(nerf_model, coords, colors, optimizer, epochs=300)

    # Image generation parameters
    img_width = 128
    img_height = 128
    focal_length = 1.0
    eye = torch.tensor([0.0, 0.0, -2.0])
    look_at = torch.tensor([0.0, 0.0, 0.0])
    up = torch.tensor([0.0, 1.0, 0.0])

    # Generate image
    generated_image = generate_image(nerf_model, img_width, img_height, focal_length, eye, look_at, up)

    plt.imsave(RESULTS_DIR / "generated_image.png", generated_image)
    print("Generated image saved as learn/Generative_Models/results/generated_image.png")

if __name__ == "__main__":
    main()
