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


def get_data(test_size=0.2, random_state=42):
    digits = load_digits()
    X = digits.data.astype(float)
    train_imgs, test_imgs = train_test_split(X, test_size=test_size, random_state=random_state, shuffle=True)
    print(f"Train and test shape are {train_imgs.shape}, {test_imgs.shape}")
    return train_imgs, test_imgs, digits.images.shape[1:]


class PCA:
    def __init__(self, components):
        self.components = components

    def _center(self, X):
        return X - self.mean_

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        X_centered = self._center(X)

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.explained_variance_ = eigenvalues[sorted_indices]
        self.components_ = eigenvectors[:, sorted_indices[: self.components]]

        total_variance = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return self._center(X) @ self.components_

    def pred(self, X):
        return self.transform(X)

    def inverse_transform(self, transformed_data):
        transformed_data = np.asarray(transformed_data, dtype=float)
        return transformed_data @ self.components_.T + self.mean_

    def pca_inverse_transform(self, transformed_data):
        return self.inverse_transform(transformed_data)


if __name__ == "__main__":
    ensure_results_dir()
    train_imgs, test_imgs, image_shape = get_data()
    img_dim = train_imgs.shape[1]
    n_components = 16

    print(f"Number of retained dimensions compared to original: {(n_components / img_dim) * 100:.2f}%")
    model = PCA(n_components)
    model.fit(train_imgs)

    transformed = model.transform(test_imgs)
    reconstructed = model.inverse_transform(transformed)

    explained_variance = float(np.sum(model.explained_variance_ratio_[:n_components]))
    print(f"Explained variance captured by first {n_components} components: {explained_variance:.4f}")

    rng = np.random.default_rng(42)
    testidx = int(rng.integers(0, len(test_imgs)))
    print(f"Test index is {testidx}")

    orgimg = test_imgs[testidx].reshape(image_shape)
    recons = np.clip(reconstructed[testidx], 0, 16).reshape(image_shape)
    print(f"Test images and reconstruction shape are {test_imgs.shape}, {reconstructed.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orgimg, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(recons, cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    fig.savefig(RESULTS_DIR / "recons.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    cumulative_variance = np.cumsum(model.explained_variance_ratio_)
    ax.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance)
    ax.set_title("Cumulative Explained Variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Explained variance ratio")
    fig.savefig(RESULTS_DIR / "variance.png", bbox_inches="tight")
    plt.close(fig)


