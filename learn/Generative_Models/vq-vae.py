from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def get_digits_loaders(batch_size=64, random_state=42):
    digits = load_digits()
    X = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
    y = torch.tensor(digits.target, dtype=torch.long)

    train_indices, test_indices = train_test_split(
        torch.arange(len(X)).numpy(),
        test_size=0.2,
        random_state=random_state,
        stratify=y.numpy(),
    )

    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    test_dataset = TensorDataset(X[test_indices], y[test_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1)
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)
        quantized = torch.matmul(encodings.float(), self.embedding.weight).reshape(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2)
        encoding_indices = encoding_indices.reshape(input_shape[0], input_shape[1], input_shape[2])
        return quantized, loss, encoding_indices


class Encoder(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=32, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim=embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, encoding_indices = self.vector_quantizer(z)
        reconstructions = self.decoder(quantized)
        return reconstructions, vq_loss, encoding_indices


def train_step(model, optimizer, images, device):
    images = images.to(device)
    optimizer.zero_grad()
    reconstructions, vq_loss, _ = model(images)
    recon_loss = F.mse_loss(reconstructions, images)
    total_loss = recon_loss + vq_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), recon_loss.item(), vq_loss.item()


def train_vqvae(model, train_loader, num_epochs, device, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (images, _) in enumerate(train_loader):
            loss, recon_loss, vq_loss = train_step(model, optimizer, images, device)
            total_train_loss += loss

            if batch_idx == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Loss: {loss:.4f}, Recon Loss: {recon_loss:.4f}, VQ Loss: {vq_loss:.4f}"
                )

        avg_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")


def inference_step(model, images, device):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        reconstructions, _, encoding_indices = model(images)
    return reconstructions.cpu(), encoding_indices.cpu()


def visualize_results(original, reconstruction, save_path, num_images=8):
    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))

    for i in range(num_images):
        axes[0, i].imshow(original[i, 0].numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        axes[1, i].imshow(reconstruction[i, 0].numpy(), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstructed")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ensure_results_dir()
    batch_size = 64
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_digits_loaders(batch_size=batch_size)
    model = VQVAE(num_embeddings=64, embedding_dim=32, commitment_cost=0.25)

    train_vqvae(model, train_loader, num_epochs, device)

    test_images, _ = next(iter(test_loader))
    reconstructions, codes = inference_step(model, test_images, device)
    print(f"Discrete code grid shape: {tuple(codes.shape)}")
    visualize_results(test_images, reconstructions, RESULTS_DIR / "vqvae_reconstructions.png")
