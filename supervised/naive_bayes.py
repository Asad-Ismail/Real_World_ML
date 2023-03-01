import numpy as np


def get_data():
    from sklearn.datasets import make_blobs
    #just to make blobs
    X, Y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X,Y

class NaiveBayes:
    def __init__(self):
        pass
    
    def prior_prob(y):
        """
        Calculate the prior probability of each class.
        
        Arguments:
        y -- a numpy array of shape (n_samples,) containing the class labels
        
        Returns:
        a numpy array of shape (n_classes,) containing the prior probability of each class
        """
        n_samples = y.shape[0]
        n_classes = len(np.unique(y))
        prior = np.zeros(n_classes)
        for i in range(n_classes):
            prior[i] = np.sum(y == i) / n_samples
        return prior
    
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



