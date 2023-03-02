import numpy as np
import matplotlib.pyplot as plt
from utils import plot_decision_boundary

def get_data():
    from sklearn.datasets import make_blobs
    #just to make blobs
    X, Y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X,Y

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        ## Just memorize all the training data for training
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for i in range(len(X)):
            distances = np.sqrt(np.sum(np.square(self.X_train - X[i,:]), axis=1))
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            y_pred[i]=np.bincount(nearest_labels).argmax()

        return y_pred

if __name__=="__main__":
    X,Y=get_data()
    #Y=np.where(Y==0,-1,1)
    print(f"Y min and max are {Y.min()},{Y.max()}")
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.savefig('results/data.png')
    knn=KNN(k=3)
    knn.fit(X, Y)
    plot_decision_boundary(knn, X, Y,save_path="results/knn.png")
    y_pred = knn.predict(X)
    accuracy = np.sum(y_pred == Y) / len(Y)
    print(f"Accuracy: {accuracy}")


