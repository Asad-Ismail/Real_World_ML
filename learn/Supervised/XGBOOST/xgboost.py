import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Simple weak learner
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions



class GradientBoosting:
    def __init__(self, n_learners=5):
        self.n_learners = n_learners
        self.learners = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, 1/n_samples)
        
        for _ in range(self.n_learners):
            learner = DecisionStump()
            min_error = float('inf')
            # Simple model based on a single feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    # Predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    # Error
                    misclassified = w[y != predictions]
                    error = sum(misclassified)
                    # Check if polarity -1 gives a better accuracy
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    # Update rule
                    if error < min_error:
                        learner.polarity = p
                        learner.threshold = threshold
                        learner.feature_index = feature_i
                        min_error = error
            # Calculate alpha
            EPS = 1e-10
            learner.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            # Update weights
            predictions = learner.predict(X)
            w *= np.exp(-learner.alpha * y * predictions)
            w /= np.sum(w)
            self.learners.append(learner)

    def predict(self, X):
        learner_preds = np.array([learner.alpha * learner.predict(X) for learner in self.learners])
        y_pred = np.sign(np.sum(learner_preds, axis=0))
        return y_pred


