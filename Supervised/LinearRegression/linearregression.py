import numpy as np

class LinearRegression:
    
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
        
    def fit(self, X, y):
        # Add a column of ones to X to represent the intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Calculate the OLS coefficients
        self.coef_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
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
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([3, 7, 11, 15, 19])

    # Create a LinearRegression object and fit the model to the data
    lr = LinearRegression()
    lr.fit(X, y)

    # Print the learned coefficients
    print("Intercept: ", lr.intercept_)
    print("Coefficients: ", lr.coef_)

    # Use the model to make predictions on new data
    X_new = np.array([[2, 3], [4, 5], [6, 7]])
    y_pred = lr.predict(X_new)
    print("Predictions: ", y_pred)
    # Calculate the residuals
    residuals = np.sqrt((y - y_pred)**2)
    # Print the residuals
    print("Residuals: ", residuals)