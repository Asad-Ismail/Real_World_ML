import numpy as np
import matplotlib.pyplot as plt



def get_data():
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X,Y


def hinge_loss(y_true, y_pred):
    return np.maximum(0, 1 - y_true * y_pred)


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=1, num_iterations=100000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
    
    def fit(self, X, y,num_epochs=100):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0

        # Training loop
        for epoch in range(num_epochs):
            for i in range(n_samples):
                # Compute the prediction and loss
                y_pred = np.dot(X[i], self.w) + self.b
                loss = hinge_loss(y[i], y_pred)

                # Compute the gradient
                if loss == 0:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - y[i] * X[i]
                    db = -y[i]

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
    
    def predict(self, X):
        y_pred = np.dot(X, self.w) + self.b
        return np.sign(y_pred)


def plot_decision_boundary(model, X, y):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.show()
    plt.savefig('results/svm.png')

  
if __name__=="__main__":

    X,Y=get_data()
    Y=np.where(Y==0,-1,1)
    print(f"Y min and max are {Y.min()},{Y.max()}")
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.savefig('results/data.png')
    svm=SVM()
    svm.fit(X, Y)
    plot_decision_boundary(svm, X, Y)
    y_pred = svm.predict(X)
    accuracy = np.sum(y_pred == Y) / len(Y)
    print(f"Accuracy: {accuracy}")


