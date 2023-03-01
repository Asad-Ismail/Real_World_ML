import numpy as np
import matplotlib.pyplot as plt
from utils import plot_decision_boundary

def get_data():
    from sklearn.datasets import make_blobs
    #just to make blobs
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
                
                # The loss is loss=hingeloss+lambda*(w**2)= w*x+b)=0+lambda*
                # If hingeloss is 0 then gradient w.rt weights is 2*lambda*W 
                # If hingeloss is 0 the gradient w.r.t bias is 0 
                # Compute the gradient
                if loss == 0:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                # loss= 1-y(w*x+b)+lambda(w**2)=2*lambda
                # If hingeloss is non zero then gradient w.rt weights is 2*lambda*W 
                # If hingeloss is 0 the gradient w.r.t bias is 0 
                else:
                    dw = 2 * self.lambda_param * self.w - y[i] * X[i]
                    db = -y[i]

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
    
    def predict(self, X):
        y_pred = np.dot(X, self.w) + self.b
        return np.sign(y_pred)



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


