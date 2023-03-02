import numpy as np
import matplotlib.pyplot as plt
from utils import plot_decision_boundary


def get_data():
    from sklearn.datasets import make_blobs
    #just to make blobs
    X, Y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X,Y

class NaiveBayes:
    
    def __init__(self):
        pass

    @staticmethod
    def mean_var(X, y):
        """
        Calculate the mean and variance of each feature for each class.
        
        Arguments:
        X -- a numpy array of shape (n_samples, n_features) containing the features
        y -- a numpy array of shape (n_samples,) containing the class labels
        
        Returns:
        a tuple (mean, var) containing two numpy arrays of shape (n_classes, n_features) 
        containing the mean and variance of each feature for each class
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        mean = np.zeros((n_classes, n_features))
        var = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            X_i = X[y == i]
            mean[i, :] = np.mean(X_i, axis=0)
            var[i, :] = np.var(X_i, axis=0)
        return mean, var

    
    def prior_prob(self,y):
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
    
    def fit(self,X,y):
        self.prior=self.prior_prob(y)
        self.mean,self.var=self.mean_var(X,y)

    def predict(self,X):
        """
        Predict the class label of a new sample using the Naive Bayes algorithm.
        
        Arguments:
        X -- a numpy array of shape (n_samples, n_features) containing the features of the new sample
        prior -- a numpy array of shape (n_classes,) containing the prior probability of each class
        mean -- a numpy array of shape (n_classes, n_features) containing the mean of each feature for each class
        var -- a numpy array of shape (n_classes, n_features) containing the variance of each feature for each class
        
        Returns:
        a numpy array of shape (n_samples,) containing the predicted class label of each sample
        """
        n_samples, n_features = X.shape
        n_classes = len(self.prior)
        likelihood = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            # calculate the class conditional probability using Gaussian distribution
            likelihood[:, i] = np.prod(1 / np.sqrt(2 * np.pi * self.var[i, :]) * np.exp(-(X - self.mean[i, :])**2 / (2 * self.var[i, :])), axis=1)
        # calculate the posterior probability and predict the class with the highest probability
        posterior = likelihood * self.prior
        preds = np.argmax(posterior,axis=1)
        return preds



if __name__=="__main__":
    X,Y=get_data()
    #Y=np.where(Y==0,-1,1)
    print(f"Y min and max are {Y.min()},{Y.max()}")
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.savefig('results/data.png')
    nb=NaiveBayes()
    nb.fit(X, Y)
    plot_decision_boundary(nb, X, Y,save_path="results/naivebayes.png")
    y_pred = nb.predict(X)
    accuracy = np.sum(y_pred == Y) / len(Y)
    print(f"Accuracy: {accuracy}")
