import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    
    def __init__(self, alpha=1.0):
        ## Alpha indicates regularizer 
        self.alpha = alpha
        self.intercept_ = None
        self.coef_ = None
        
    def fit(self, X, y):
        # Add a column of ones to X to represent the intercept term
        X = np.hstack((np.ones((X.shape[0],1)), X))
        
        # Calculate the ridge regression coefficients
        I = np.eye(X.shape[1])
        I[0,0] = 0 # Don't regularize the intercept
        self.coef_ = np.linalg.inv(X.T.dot(X) + self.alpha*I).dot(X.T).dot(y)
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
        
    def predict(self, X):
        # Add a column of ones to X to represent the intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Predict y using the learned coefficients
        y_pred = X.dot(np.hstack((self.intercept_, self.coef_)))
        return y_pred

if __name__=="__main__":
    # Create a toy dataset with 2d feature vector
    X = np.array([1, 3, 5, 7, 9, 10])
    X=X.reshape(-1,1)
    y = np.array([3, 7, 11, 15, 19,23])
    mean = 0
    std_dev = 0.5
    # create a noise array with the same shape as y
    noise = np.random.normal(mean, std_dev, size=y.shape)
    y=y+noise
    plt.figure()
    plt.scatter(X,y,color='g')
    plt.savefig('results/data.png')

    # Create a LinearRegression object and fit the model to the data
    lr = LinearRegression()
    lr.fit(X, y)

    # Print the learned coefficients
    print("Intercept: ", lr.intercept_)
    print("Coefficients: ", lr.coef_)

    # Use the model to make predictions on new data
    #X_new = np.array([[2, 3], [4, 5], [6, 7]])
    y_pred = lr.predict(X)
    print("Predictions: ", y_pred)
    # Calculate the residuals
    residuals = np.mean(np.sqrt((y - y_pred)**2))
    # Print the residuals
    print("Mean Residuals: ", residuals)
    plt.scatter(X,y_pred,color='r')
    plt.savefig('results/linear_regression.png')