from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def forward(self, x):
        stats = self.net(x)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class TextDecoder(nn.Module):
    def __init__(self, latent_dim, vocab_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, vocab_size),
        )

    def forward(self, z):
        return self.net(z)


class MultimodalVAE(nn.Module):
    """
    Lightweight multimodal VAE concept demo.

    The image modality uses digit images. The text modality uses a bag-of-words
    encoding of the corresponding digit word label.
    """

    def __init__(self, image_dim, vocab_size, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_encoder = MLPEncoder(image_dim, 128, latent_dim)
        self.text_encoder = MLPEncoder(vocab_size, 64, latent_dim)
        self.image_decoder = ImageDecoder(latent_dim, image_dim)
        self.text_decoder = TextDecoder(latent_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def product_of_experts(self, image_stats, text_stats):
        image_mu, image_logvar = image_stats
        text_mu, text_logvar = text_stats

        image_precision = torch.exp(-image_logvar)
        text_precision = torch.exp(-text_logvar)

        combined_precision = image_precision + text_precision
        combined_mu = (image_mu * image_precision + text_mu * text_precision) / combined_precision
        combined_logvar = torch.log(1.0 / combined_precision)
        return combined_mu, combined_logvar

    def forward(self, images, texts):
        image_mu, image_logvar = self.image_encoder(images)
        text_mu, text_logvar = self.text_encoder(texts)
        combined_mu, combined_logvar = self.product_of_experts((image_mu, image_logvar), (text_mu, text_logvar))
        z = self.reparameterize(combined_mu, combined_logvar)
        image_recon = self.image_decoder(z)
        text_logits = self.text_decoder(z)
        return image_recon, text_logits, combined_mu, combined_logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        image_recon = self.image_decoder(z)
        text_logits = self.text_decoder(z)
        return image_recon, text_logits


def load_multimodal_data(test_size=0.2, random_state=42):
    digits = load_digits()
    images = torch.tensor(digits.data, dtype=torch.float32) / 16.0
    labels = torch.tensor(digits.target, dtype=torch.long)
    texts = F.one_hot(labels, num_classes=10).float()

    train_idx, test_idx = train_test_split(
        torch.arange(len(images)).numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=labels.numpy(),
    )

    train_dataset = TensorDataset(images[train_idx], texts[train_idx], labels[train_idx])
    test_dataset = TensorDataset(images[test_idx], texts[test_idx], labels[test_idx])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_loader, test_loader


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def train(model, data_loader, optimizer, epochs, device):
    history = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, texts, labels in data_loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            image_recon, text_logits, mu, logvar = model(images, texts)

            image_loss = F.mse_loss(image_recon, images)
            text_loss = F.cross_entropy(text_logits, labels)
            kl_loss = kl_divergence(mu, logvar)
            loss = image_loss + text_loss + 0.1 * kl_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return history


def decode_text_predictions(text_logits):
    predictions = torch.argmax(text_logits, dim=1).cpu().tolist()
    return [DIGIT_WORDS[index] for index in predictions]


if __name__ == "__main__":
    ensure_results_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_multimodal_data()

    model = MultimodalVAE(image_dim=64, vocab_size=10, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = train(model, train_loader, optimizer, epochs=25, device=device)

    test_images, test_texts, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)
    test_texts = test_texts.to(device)

    model.eval()
    with torch.no_grad():
        image_recon, text_logits, _, _ = model(test_images, test_texts)
        sampled_images, sampled_text_logits = model.sample(num_samples=5, device=device)

    predicted_words = decode_text_predictions(text_logits[:5])
    sampled_words = decode_text_predictions(sampled_text_logits)

    print("Predicted text labels for the first five test images:", predicted_words)
    print("Sampled text labels from random latent vectors:", sampled_words)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for idx in range(5):
        axes[0, idx].imshow(test_images[idx].cpu().reshape(8, 8), cmap="gray")
        axes[0, idx].set_title(f"True: {DIGIT_WORDS[test_labels[idx].item()]}")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(image_recon[idx].cpu().reshape(8, 8), cmap="gray")
        axes[1, idx].set_title(f"Pred: {predicted_words[idx]}")
        axes[1, idx].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "multimodal_recon.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for idx in range(5):
        axes[idx].imshow(sampled_images[idx].cpu().reshape(8, 8), cmap="gray")
        axes[idx].set_title(sampled_words[idx])
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "multimodal_samples.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(history) + 1), history)
    ax.set_title("Multimodal VAE Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(RESULTS_DIR / "multimodal_loss.png", bbox_inches="tight")
    plt.close(fig)
