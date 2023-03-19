import torch
import torch.nn as nn
import numpy as np

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
    camera_origin = eye
    camera_direction = (look_at - eye).unsqueeze(-1).float()
    camera_direction /= torch.norm(camera_direction)
    camera_right = torch.cross(camera_direction.squeeze(-1), up).unsqueeze(-1).float()
    camera_right /= torch.norm(camera_right)
    camera_up = torch.cross(camera_right.squeeze(-1), camera_direction.squeeze(-1)).unsqueeze(-1).float()

    x = torch.linspace(-aspect_ratio, aspect_ratio, img_width)
    y = torch.linspace(-1.0, 1.0, img_height)
    xv, yv = torch.meshgrid(x, y)

    coords = torch.stack([xv, yv], dim=-1).view(-1, 2)
    coords = torch.cat([coords, torch.ones(coords.shape[0], 1) * focal_length], dim=-1)

    world_coords = (coords[..., 0:1] * camera_right + coords[..., 1:2] * camera_up + coords[..., 2:3] * camera_direction).T
    world_coords = world_coords + camera_origin.unsqueeze(-1)
    world_coords = world_coords.view(3, img_height, img_width)

    colors = nerf_model(world_coords.view(3, -1).T)
    colors = colors.view(img_height, img_width, 3).detach().numpy()

    return np.clip(colors, 0, 1)

def main():
    input_dim = 3
    hidden_dim = 256
    output_dim = 3
    num_layers = 4

    nerf_model = MLP(input_dim, hidden_dim, output_dim, num_layers)
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=1e-3)

    # Prepare your training data: coordinates (3D points) and colors (RGB)
    coords = torch.randn(1000, 3)
    colors = torch.randn(1000, 3)

    train_nerf(nerf_model, coords, colors, optimizer, epochs=1000
    train_nerf(nerf_model, coords, colors, optimizer, epochs=1000)

    # Image generation parameters
    img_width = 512
    img_height = 512
    focal_length = 1.0
    eye = torch.tensor([0.0, 0.0, -2.0])
    look_at = torch.tensor([0.0, 0.0, 0.0])
    up = torch.tensor([0.0, 1.0, 0.0])

    # Generate image
    generated_image = generate_image(nerf_model, img_width, img_height, focal_length, eye, look_at, up)

    # Save generated image
    import matplotlib.pyplot as plt
    plt.imsave('generated_image.png', generated_image)
    print("Generated image saved as 'generated_image.png'")

if __name__ == "__main__":
    main()
