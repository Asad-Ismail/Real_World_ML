from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


class Encoder:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
        self.projection_ = vt[: self.latent_dim].T
        return self

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.projection_


class Codebook:
    def __init__(self, num_codes, max_iter=25, random_state=42):
        self.num_codes = num_codes
        self.max_iter = max_iter
        self.random_state = random_state

    def assign(self, latents, embeddings=None):
        embeddings = self.embeddings_ if embeddings is None else embeddings
        distances = np.linalg.norm(latents[:, None, :] - embeddings[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def fit(self, latents):
        rng = np.random.default_rng(self.random_state)
        embeddings = latents[rng.choice(latents.shape[0], self.num_codes, replace=False)].copy()

        for _ in range(self.max_iter):
            assignments = self.assign(latents, embeddings)
            new_embeddings = embeddings.copy()

            for code_index in range(self.num_codes):
                members = latents[assignments == code_index]
                if members.size == 0:
                    new_embeddings[code_index] = latents[rng.integers(0, latents.shape[0])]
                else:
                    new_embeddings[code_index] = np.mean(members, axis=0)

            if np.allclose(new_embeddings, embeddings):
                break
            embeddings = new_embeddings

        self.embeddings_ = embeddings
        return self

    def __call__(self, latents):
        assignments = self.assign(latents)
        quantized = self.embeddings_[assignments]
        return quantized, assignments


class Decoder:
    def fit(self, latents, targets):
        latents = np.asarray(latents, dtype=float)
        targets = np.asarray(targets, dtype=float)
        design = np.hstack([latents, np.ones((latents.shape[0], 1))])
        self.weights_ = np.linalg.pinv(design) @ targets
        return self

    def __call__(self, latents):
        latents = np.asarray(latents, dtype=float)
        if latents.ndim == 1:
            latents = latents[None, :]
        design = np.hstack([latents, np.ones((latents.shape[0], 1))])
        return design @ self.weights_


class Discriminator:
    def __init__(self, lr=0.2, num_steps=300):
        self.lr = lr
        self.num_steps = num_steps

    def fit(self, real_images, fake_images):
        X = np.vstack([real_images, fake_images])
        y = np.concatenate([np.ones(real_images.shape[0]), np.zeros(fake_images.shape[0])])

        self.weights_ = np.zeros(X.shape[1], dtype=float)
        self.bias_ = 0.0

        for _ in range(self.num_steps):
            logits = X @ self.weights_ + self.bias_
            probs = sigmoid(logits)
            error = probs - y
            self.weights_ -= self.lr * (X.T @ error) / X.shape[0]
            self.bias_ -= self.lr * np.mean(error)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        return sigmoid(X @ self.weights_ + self.bias_)


class VQGAN:
    """
    Compact VQ-GAN-style concept demo.

    This file illustrates the main pieces:
    - an encoder that maps images to a latent space
    - a codebook that quantizes those latent vectors
    - a decoder that reconstructs from codebook entries
    - a discriminator that scores real vs reconstructed images

    It is not a full neural VQ-GAN implementation. The goal is to make the
    ideas executable and easy to inspect with lightweight dependencies.
    """

    def __init__(self, input_dim, latent_dim=12, codebook_size=32, codebook_iters=25):
        self.encoder = Encoder(latent_dim)
        self.codebook = Codebook(codebook_size, max_iter=codebook_iters)
        self.decoder = Decoder()
        self.discriminator = Discriminator()
        self.input_dim = input_dim

    def fit(self, X):
        self.encoder.fit(X)
        z_e = self.encoder(X)
        self.codebook.fit(z_e)
        z_q, assignments = self.codebook(z_e)
        self.decoder.fit(z_q, X)
        reconstructions = np.clip(self.decoder(z_q), 0.0, 1.0)
        self.discriminator.fit(X, reconstructions)

        metrics = {
            "reconstruction_mse": float(np.mean((X - reconstructions) ** 2)),
            "quantization_error": float(np.mean((z_e - z_q) ** 2)),
            "mean_real_score": float(np.mean(self.discriminator.score(X))),
            "mean_fake_score": float(np.mean(self.discriminator.score(reconstructions))),
            "num_codes_used": int(np.unique(assignments).size),
        }
        return metrics

    def forward(self, X):
        z_e = self.encoder(X)
        z_q, assignments = self.codebook(z_e)
        reconstructions = np.clip(self.decoder(z_q), 0.0, 1.0)
        return reconstructions, z_e, z_q, assignments

    def generate(self, num_samples, seed=42):
        rng = np.random.default_rng(seed)
        latent_codes = self.codebook.embeddings_[rng.integers(0, self.codebook.num_codes, size=num_samples)]
        return np.clip(self.decoder(latent_codes), 0.0, 1.0)


def load_data(test_size=0.2, random_state=42):
    digits = load_digits()
    X = digits.data.astype(float) / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, digits.images.shape[1:]


def plot_pairs(originals, reconstructions, image_shape, save_path):
    n_samples = len(originals)
    fig, axes = plt.subplots(2, n_samples, figsize=(1.8 * n_samples, 4))
    for idx in range(n_samples):
        axes[0, idx].imshow(originals[idx].reshape(image_shape), cmap="gray")
        axes[0, idx].axis("off")
        axes[0, idx].set_title("Original")
        axes[1, idx].imshow(reconstructions[idx].reshape(image_shape), cmap="gray")
        axes[1, idx].axis("off")
        axes[1, idx].set_title("Recon")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_grid(images, image_shape, save_path, title, n_cols=5):
    n_samples = len(images)
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 1.8 * n_rows))
    axes = np.atleast_2d(axes)

    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index < n_samples:
            ax.imshow(images[index].reshape(image_shape), cmap="gray")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ensure_results_dir()
    X_train, X_test, _, _, image_shape = load_data()

    model = VQGAN(input_dim=X_train.shape[1], latent_dim=12, codebook_size=32, codebook_iters=25)
    metrics = model.fit(X_train)
    reconstructions, _, _, _ = model.forward(X_test[:10])
    generated_images = model.generate(10)

    print("VQ-GAN concept demo metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    plot_pairs(X_test[:10], reconstructions, image_shape, RESULTS_DIR / "vqgan_reconstructions.png")
    plot_grid(generated_images, image_shape, RESULTS_DIR / "vqgan_generated.png", "Generated Samples")
