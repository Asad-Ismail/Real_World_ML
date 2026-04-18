from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_decision_boundary(model, X, y, save_path="results/svm.png", show=False):
    save_path = Path(save_path)
    ensure_directory(save_path.parent)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.1),
        np.arange(x2_min, x2_max, 0.1),
    )
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx1, xx2, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
