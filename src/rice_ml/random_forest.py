import numpy as np
from collections import Counter
from .decision_trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


class RandomForestBase:
    """
    Base class for Random Forest models.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_ = []
        self.n_features_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _bootstrap_sample(self, X, y):
        """
        Draw a bootstrap sample from the dataset.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _get_max_features(self):
        """
        Determine number of features to consider at each split.
        """
        if self.max_features == "sqrt":
            return int(np.sqrt(self.n_features_))
        elif self.max_features == "log2":
            return int(np.log2(self.n_features_))
        elif isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return int(self.max_features * self.n_features_)
        else:
            return self.n_features_


class RandomForestClassifier(RandomForestBase):
    """
    Random Forest Classifier.
    """

    def fit(self, X, y):
        """
        Fit Random Forest classifier.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features_ = X.shape[1]
        self.trees_ = []

        max_features = self._get_max_features()

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        """
        Predict class labels using majority voting.
        """
        X = np.asarray(X)

        # Collect predictions from all trees
        all_preds = np.array([tree.predict(X) for tree in self.trees_])

        # Majority vote
        y_pred = []
        for i in range(X.shape[0]):
            votes = Counter(all_preds[:, i])
            y_pred.append(votes.most_common(1)[0][0])

        return np.array(y_pred)


class RandomForestRegressor(RandomForestBase):
    """
    Random Forest Regressor.
    """

    def fit(self, X, y):
        """
        Fit Random Forest regressor.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features_ = X.shape[1]
        self.trees_ = []

        max_features = self._get_max_features()

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        """
        Predict continuous values using averaging.
        """
        X = np.asarray(X)

        # Average predictions from all trees
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(all_preds, axis=0)
