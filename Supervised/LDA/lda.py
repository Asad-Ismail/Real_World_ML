import numpy as np

class LDA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Compute the overall mean
        mean_overall = np.mean(X, axis=0)

        # Compute the within-class scatter matrix and the between-class scatter matrix
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)

            # Within-class scatter matrix
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            # Between-class scatter matrix
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * mean_diff.dot(mean_diff.T)

        # Compute the eigenvectors and eigenvalues of the matrix inv(S_W).dot(S_B)
        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Sort eigenvectors by decreasing eigenvalues and select the top n_components
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.linear_discriminants = eigenvectors[:self.n_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        raise NotImplementedError("LDA is a dimensionality reduction method and does not support 'predict' method. Use the transformed data with another classifier for predictions.")




from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Train a classifier on the transformed data
clf = LogisticRegression(random_state=42)
clf.fit(X_train_lda, y_train)

# Make predictions
y_pred = clf.predict(X_test_lda)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
