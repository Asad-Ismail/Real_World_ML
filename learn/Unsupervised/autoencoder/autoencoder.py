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


class Autoencoder:
    def __init__(self, input_size, hidden_size, learning_rate=0.05, num_epochs=400, random_state=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.random_state = random_state

        rng = np.random.default_rng(random_state)
        self.W_encoder = rng.normal(scale=0.05, size=(input_size, hidden_size))
        self.b_encoder = np.zeros(hidden_size, dtype=float)
        self.W_decoder = rng.normal(scale=0.05, size=(hidden_size, input_size))
        self.b_decoder = np.zeros(input_size, dtype=float)

    def _forward(self, X):
        hidden_linear = X @ self.W_encoder + self.b_encoder
        hidden = np.maximum(hidden_linear, 0.0)
        output_linear = hidden @ self.W_decoder + self.b_decoder
        reconstruction = sigmoid(output_linear)
        return hidden_linear, hidden, output_linear, reconstruction

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.loss_history_ = []
        n_samples = X.shape[0]

        for epoch in range(self.num_epochs):
            hidden_linear, hidden, _, reconstruction = self._forward(X)
            error = reconstruction - X
            loss = float(np.mean((reconstruction - X) ** 2))
            self.loss_history_.append(loss)

            d_output = (2.0 / n_samples) * error * reconstruction * (1.0 - reconstruction)
            grad_W_decoder = hidden.T @ d_output
            grad_b_decoder = np.sum(d_output, axis=0)

            d_hidden = d_output @ self.W_decoder.T
            d_hidden_linear = d_hidden * (hidden_linear > 0.0)
            grad_W_encoder = X.T @ d_hidden_linear
            grad_b_encoder = np.sum(d_hidden_linear, axis=0)

            self.W_decoder -= self.learning_rate * grad_W_decoder
            self.b_decoder -= self.learning_rate * grad_b_decoder
            self.W_encoder -= self.learning_rate * grad_W_encoder
            self.b_encoder -= self.learning_rate * grad_b_encoder

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss:.6f}")
        return self

    def encode(self, X):
        X = np.asarray(X, dtype=float)
        hidden_linear = X @ self.W_encoder + self.b_encoder
        return np.maximum(hidden_linear, 0.0)

    def decode(self, Z):
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[None, :]
        output_linear = Z @ self.W_decoder + self.b_decoder
        return sigmoid(output_linear)

    def reconstruct(self, X):
        return self.decode(self.encode(X))


def load_data(test_size=0.2, random_state=42):
    digits = load_digits()
    X = digits.data.astype(float) / 16.0
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state, shuffle=True)
    return X_train, X_test, digits.images.shape[1:]


if __name__ == "__main__":
    ensure_results_dir()
    X_train, X_test, image_shape = load_data()

    input_size = X_train.shape[1]
    hidden_size = 16
    model = Autoencoder(input_size, hidden_size, learning_rate=0.05, num_epochs=400, random_state=42)

    print(f"Number of latent dimensions compared to original: {(hidden_size / input_size) * 100:.2f}%")
    model.fit(X_train)

    testidx = 7
    print(f"Test index is {testidx}")
    original = X_test[testidx].reshape(image_shape)
    reconstruction = model.reconstruct(X_test[testidx])[0].reshape(image_shape)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(reconstruction, cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    fig.savefig(RESULTS_DIR / "recons.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(model.loss_history_) + 1), model.loss_history_)
    ax.set_title("Autoencoder Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    fig.savefig(RESULTS_DIR / "loss.png", bbox_inches="tight")
    plt.close(fig)
