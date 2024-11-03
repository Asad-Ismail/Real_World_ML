from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Linear Regression Example (Interpretable)
X = np.array([[1], [2], [3], [4], [5]])  # Simple dataset
y = np.array([2, 4, 6, 8, 10])  # Directly proportional relationship
model = LinearRegression()
model.fit(X, y)
print("Linear Regression Coefficients:", model.coef_)

# Decision Tree Example for Explainability
iris = load_iris()
X, y = iris.data, iris.target
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X, y)
print("Feature importances:", tree_model.feature_importances_)

# Visualizing the feature importances
plt.barh(range(iris.data.shape[1]), tree_model.feature_importances_)
plt.yticks(np.arange(iris.data.shape[1]), iris.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
