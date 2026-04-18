from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from utils import ensure_directory, plot_decision_boundary


def get_data():
    from sklearn.datasets import make_blobs
    #just to make blobs
    X, Y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X,Y

class NaiveBayes:
    
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.prior = np.zeros(n_classes, dtype=float)
        self.mean = np.zeros((n_classes, n_features), dtype=float)
        self.var = np.zeros((n_classes, n_features), dtype=float)

        for class_index, class_label in enumerate(self.classes_):
            X_class = X[y == class_label]
            self.prior[class_index] = X_class.shape[0] / X.shape[0]
            self.mean[class_index] = np.mean(X_class, axis=0)
            self.var[class_index] = np.var(X_class, axis=0) + self.var_smoothing
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        joint_log_likelihood = []

        for class_index, _ in enumerate(self.classes_):
            log_prior = np.log(self.prior[class_index])
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.var[class_index])
                + ((X - self.mean[class_index]) ** 2) / self.var[class_index],
                axis=1,
            )
            joint_log_likelihood.append(log_prior + log_likelihood)

        return np.column_stack(joint_log_likelihood)

    def predict(self, X):
        posterior = self._joint_log_likelihood(X)
        class_indices = np.argmax(posterior, axis=1)
        return self.classes_[class_indices]



if __name__=="__main__":
    X,Y=get_data()
    print(f"Y min and max are {Y.min()},{Y.max()}")
    results_dir = ensure_directory(CURRENT_DIR / "results")

    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.savefig(results_dir / "data.png")
    plt.close()

    nb=NaiveBayes()
    nb.fit(X, Y)
    plot_decision_boundary(nb, X, Y, save_path=results_dir / "naivebayes.png")
    y_pred = nb.predict(X)
    accuracy = np.sum(y_pred == Y) / len(Y)
    print(f"Accuracy: {accuracy}")
