# coding:utf-8
import numpy as np

try:
    from .basemodel.base import BaseEstimator
    from .base import information_gain, mse_criterion
    from .tree import Tree
except ImportError:
    from basemodel.base import BaseEstimator
    from base import information_gain, mse_criterion
    from tree import Tree


class RandomForest(BaseEstimator):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion=None):
        """Base class for RandomForest.

        Parameters
        ----------
        n_estimators : int
            The number of decision tree.
        max_features : int
            The number of features to consider when looking for the best split.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            Maximum depth of the tree.
        criterion : str
            The function to measure the quality of a split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = max(1, int(np.sqrt(X.shape[1])))
        elif self.max_features > X.shape[1]:
            raise ValueError("max_features cannot exceed the number of input features.")

        self._train()
        return self

    def _bootstrap_sample(self):
        indices = np.random.choice(self.n_samples, size=self.n_samples, replace=True)
        return self.X[indices], self.y[indices]

    def _train(self):
        for tree in self.trees:
            sample_X, sample_y = self._bootstrap_sample()
            tree.train(
                sample_X,
                sample_y,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )

    def _predict(self, X=None):
        raise NotImplementedError()


class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion="entropy"):
        super(RandomForestClassifier, self).__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            criterion=criterion,
        )

        if criterion == "entropy":
            self.criterion = information_gain
        else:
            raise ValueError()

        # Initialize empty trees
        for _ in range(self.n_estimators):
            self.trees.append(Tree(criterion=self.criterion))

    def fit(self, X, y):
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        return super(RandomForestClassifier, self).fit(X, encoded_y)

    def predict_proba(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is None:
            raise ValueError("You must call `fit` before `predict_proba`")

        return self._predict_proba(X)

    def _predict_proba(self, X=None):
        n_classes = len(self.classes_)
        predictions = np.zeros((X.shape[0], n_classes))

        for i in range(X.shape[0]):
            row_pred = np.zeros(n_classes)
            for tree in self.trees:
                row_pred += tree.predict_row(X[i, :])

            row_pred /= self.n_estimators
            predictions[i, :] = row_pred
        return predictions

    def _predict(self, X=None):
        probabilities = self._predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class RandomForestRegressor(RandomForest):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None, criterion="mse"):
        super(RandomForestRegressor, self).__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
        )

        if criterion == "mse":
            self.criterion = mse_criterion
        else:
            raise ValueError()

        # Initialize empty regression trees
        for _ in range(self.n_estimators):
            self.trees.append(Tree(regression=True, criterion=self.criterion))

    def _predict(self, X=None):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return predictions.mean(axis=1)
