import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import stats
sys.path.append("../utils")
from utils import plot_decision_boundary

def get_data():
    from sklearn.datasets import make_blobs
    #just to make blobs
    X, Y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X,Y

class Node:
    def __init__(self, feature=None, threshold=None, value=None, left_subtree=None, right_subtree=None):
        self.feature = feature          # index of feature to split on
        self.threshold = threshold      # threshold to split feature on
        self.value = value              # value of leaf node (if leaf node)
        self.left_subtree = left_subtree    # left subtree (if internal node)
        self.right_subtree = right_subtree  # right subtree (if internal node)

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.tree
            while node.left_subtree:
                if sample[node.feature] <= node.threshold:
                    node = node.left_subtree
                else:
                    node = node.right_subtree
            predictions.append(node.value)
        return np.array(predictions)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1:
            # Leaf node
            return Node(value=self._most_common_label(y))
        
        # Splitting the node
        feature_indices = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        
        # Grow subtrees
        left_subtree = self._grow_tree(X[left_indices, :], y[left_indices], depth+1)
        right_subtree = self._grow_tree(X[right_indices, :], y[right_indices], depth+1)
        
        # Return node
        return Node(feature=best_feature, threshold=best_threshold,
                    left_subtree=left_subtree, right_subtree=right_subtree)
    
    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        split_index, split_threshold = None, None
        for i in feature_indices:
            feature_values = X[:, i]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                gain = self._information_gain(y, feature_values, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_index = i
                    split_threshold = threshold
        return split_index, split_threshold
    
    def _split(self, feature_values, threshold):
        left_indices = np.argwhere(feature_values <= threshold).flatten()
        right_indices = np.argwhere(feature_values > threshold).flatten()
        return left_indices, right_indices

    
    def _most_common_label(self,y):
         m = stats.mode(y)[0][0]
         return m

    
    def _information_gain(self, y, feature_values, threshold):
        parent_entropy = self._entropy(y)
        
        # Calculate entropy of left and right children
        left_indices, right_indices = self._split(feature_values, threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_indices), len(right_indices)
        entropy_l = self._entropy(y[left_indices]) * n_l / n
        entropy_r = self._entropy(y[right_indices]) * n_r / n
        
        # Calculate information gain
        information_gain = parent_entropy - (entropy_l + entropy_r)
        return information_gain
    
    def _entropy(self, y):
        n = len(y)
        if n == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / n
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy


if __name__=="__main__":
    X,Y=get_data()
    print(f"Y min and max are {Y.min()},{Y.max()}")
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.savefig('results/data.png')
    dt=DecisionTree(max_depth=10)
    print(f"Fitting Training data to Decision Tree")
    dt.fit(X, Y)
    plot_decision_boundary(dt, X, Y,save_path="results/decisiontree.png")
    y_pred = dt.predict(X)
    accuracy = np.sum(y_pred == Y) / len(Y)
    print(f"Decision Tree Accuracy: {accuracy}")
