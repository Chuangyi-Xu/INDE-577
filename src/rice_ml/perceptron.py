# src/rice_ml/perceptron.py
import numpy as np

class Perceptron:
    """
    Classic Rosenblatt Perceptron binary classifier.
    Follows scikit-learn-style API: fit, predict, score.

    Parameters
    ----------
    max_iter : int
        Maximum number of passes over the training dataset.
    lr : float
        Learning rate for weight updates.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, max_iter=1000, lr=1.0, random_state=None):
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state

        self.w_ = None   # weights
        self.b_ = None   # bias

    def fit(self, X, y):
        """
        Fit the perceptron model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
            Labels should be either {0,1} or {-1,+1}.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Convert labels to {-1, +1} if needed
        y_unique = np.unique(y)
        if set(y_unique) == {0, 1}:
            y = np.where(y == 0, -1, 1)

        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = 0.0

        for _ in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y):
                if yi * (np.dot(self.w_, xi) + self.b_) <= 0:
                    # Misclassified → update
                    self.w_ += self.lr * yi * xi
                    self.b_ += self.lr * yi
                    errors += 1
            if errors == 0:
                break
        return self

    def decision_function(self, X):
        X = np.asarray(X)
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        scores = self.decision_function(X)
        # map sign to {0,1}
        labels = np.where(scores >= 0, 1, -1)
        return np.where(labels == -1, 0, 1)

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
