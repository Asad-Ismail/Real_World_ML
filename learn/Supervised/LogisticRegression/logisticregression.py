import numpy as np
import matplotlib.pyplot as plt
import os,sys
sys.path.append("../utils")
from utils import plot_decision_boundary

def get_data():
    from sklearn.datasets import make_blobs
    #just to make blobs
    X, Y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X,Y

class LogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0
        
        for i in range(self.num_iterations):
            # forward propagation
            z = np.dot(X, self.weights) + self.bias
            a = self.sigmoid(z)
            
            # backward propagation
            dz = a - y.reshape(m, 1)
            dw = np.dot(X.T, dz) / m
            db = np.sum(dz) / m
            #print(dw)
            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        a = self.sigmoid(z)
        return np.int0(np.round(a)).squeeze(1)


if __name__=="__main__":
    X,Y=get_data()
    print(f"Y min and max are {Y.min()},{Y.max()}")
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.savefig('results/data.png')
    model=LogisticRegression()
    model.fit(X, Y)
    plot_decision_boundary(model, X, Y,save_path="results/logistic_regression.png")
    y_pred = model.predict(X)
    accuracy = (np.sum(y_pred == Y)) / len(Y)
    print(f"Accuracy: {accuracy}")
