import numpy as np
import matplotlib.pyplot as plt



def get_data():
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X,Y


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=100000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient descent optimization
        for i in range(self.num_iterations):
            # Randomly select a sample
            random_index = np.random.randint(n_samples)
            sample_x, sample_y = X[random_index], y[random_index]
            
            # Calculate hinge loss and gradient
            condition = sample_y * (np.dot(sample_x, self.w) - self.b) >= 1
            if condition:
                dw = 2 * self.lambda_param * self.w
                db = 0
            else:
                dw = 2 * self.lambda_param * self.w - np.dot(sample_x, sample_y)
                db = -sample_y
            
            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
    
    def predict(self, X):
        y_pred = np.dot(X, self.w) - self.b
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

  

X,Y=get_data()
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.savefig('results/data.png')
#w,b=svm(X,Y)
# Train an SVM on the dataset
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0)
#svm=SVM()
svm.fit(X, Y)
# Extract the learned weights and bias
w = svm.w
b = svm.b
print(w)
print(b)
plot_decision_boundary(svm, X, Y)

