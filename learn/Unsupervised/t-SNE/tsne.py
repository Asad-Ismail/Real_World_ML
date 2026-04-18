from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from tsne2 import momentum_func, tsne

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "results"


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def get_data(n_samples=600):
    data, labels = load_digits(return_X_y=True)
    return data[:n_samples], labels[:n_samples]


if __name__ == "__main__":
    ensure_results_dir()
    train_imgs, train_lbls = get_data()

    unique_values, unique_counts = np.unique(train_lbls, return_counts=True)
    for value, count in zip(unique_values, unique_counts):
        print(f"Label {value} has count: {count}")

    embedding = tsne(
        train_imgs,
        n_components=2,
        perp=30,
        n_iter=300,
        lr=100,
        momentum_fn=momentum_func,
        pbar=True,
        random_state=42,
    )
    np.save(RESULTS_DIR / "tsne_wrapper.npy", embedding)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=train_lbls, cmap="tab10", s=10)
    plt.title("t-SNE Wrapper Example")
    plt.savefig(RESULTS_DIR / "clusters_wrapper.png", bbox_inches="tight")
    plt.close()
