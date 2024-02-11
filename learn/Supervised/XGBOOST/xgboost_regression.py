import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X, y = make_regression(n_samples=100, n_features=20, n_informative=15, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleRegressionTree:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        # Initialize variables to track the best split
        m = X.shape[1]
        best_mse = np.inf
        best_feature = None
        best_threshold = None

        for feature_i in range(m):
            thresholds = np.unique(X[:, feature_i])
            for threshold in thresholds:
                left_idx = X[:, feature_i] <= threshold
                right_idx = X[:, feature_i] > threshold
                if np.any(left_idx) and np.any(right_idx):
                    left_value = np.mean(y[left_idx])
                    right_value = np.mean(y[right_idx])
                    mse = self._calculate_mse(y[left_idx], left_value) + self._calculate_mse(y[right_idx], right_value)
                    if mse < best_mse:
                        best_mse = mse
                        best_feature = feature_i
                        best_threshold = threshold
                        self.left_value = left_value
                        self.right_value = right_value

        self.feature_index = best_feature
        self.threshold = best_threshold

    def _calculate_mse(self, y_true, constant):
        return np.mean((y_true - constant) ** 2)

    def predict(self, X):
        if self.feature_index is None:
            return np.zeros(X.shape[0])
        return np.where(X[:, self.feature_index] <= self.threshold, self.left_value, self.right_value)


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X, y):
        # Initial prediction is the mean of y
        self.initial_prediction = np.mean(y)  # Modify this line
        predictions = np.full(y.shape, self.initial_prediction)
        for _ in range(self.n_estimators):
            residuals = y - predictions
            tree = SimpleRegressionTree()
            tree.fit(X, residuals)
            predictions += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.full(X.shape[0], np.mean(y_train))
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions


model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"XGboost Mean Squared Error: {mse}")

mean_mse= mean_squared_error(y_test, np.array(np.mean(y_train)).repeat(y_test.shape[0]))
print(f"Mean predictor Mean Squared Error: {mean_mse}")
