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

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        y = np.asarray(y, dtype=float).reshape(m)
        self.weights = np.zeros(n, dtype=float)
        self.bias = 0
        
        for _ in range(self.num_iterations):
            # forward propagation
            z = np.dot(X, self.weights) + self.bias
            a = self.sigmoid(z)
            
            # backward propagation
            dz = a - y
            dw = np.dot(X.T, dz) / m
            db = np.sum(dz) / m

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self
        
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        a = self.sigmoid(z)
        return np.rint(a).astype(int)


if __name__=="__main__":
    X,Y=get_data()
    print(f"Y min and max are {Y.min()},{Y.max()}")
    results_dir = ensure_directory(CURRENT_DIR / "results")

    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.savefig(results_dir / "data.png")
    plt.close()

    model=LogisticRegression()
    model.fit(X, Y)
    plot_decision_boundary(model, X, Y, save_path=results_dir / "logistic_regression.png")
    y_pred = model.predict(X)
    accuracy = (np.sum(y_pred == Y)) / len(Y)
    print(f"Accuracy: {accuracy}")
