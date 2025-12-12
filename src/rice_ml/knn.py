import numpy as np
from typing import Optional


class KNNClassifier:
    """
    A simple implementation of the k-Nearest Neighbors (kNN) classifier.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider. Must be >= 1.
    metric : str, default="euclidean"
        Distance metric to use. Currently supports:
        - "euclidean"
        - "manhattan"

    Attributes
    ----------
    X_train_ : np.ndarray
        Training feature matrix of shape (n_samples, n_features).
    y_train_ : np.ndarray
        Training labels.
    """

    def __init__(self, k: int = 3, metric: str = "euclidean"):
        if k < 1:
            raise ValueError("k must be at least 1.")

        if metric not in ["euclidean", "manhattan"]:
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")

        self.k = k
        self.metric = metric

        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    # --------------------------------------------------------
    # Distance computation
    # --------------------------------------------------------
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between X and training data.

        Parameters
        ----------
        X : np.ndarray, shape (m_samples, n_features)

        Returns
        -------
        distances : np.ndarray, shape (m_samples, n_train)
            Distance matrix between each test sample and each training sample.
        """
        if self.metric == "euclidean":
            # ||X - X_train||^2 = X^2 + X_train^2 - 2 X·X_train
            X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
            train_sq = np.sum(self.X_train_ ** 2, axis=1).reshape(1, -1)
            cross = X @ self.X_train_.T
            distances = np.sqrt(X_sq + train_sq - 2 * cross)
            return distances

        elif self.metric == "manhattan":
            distances = np.sum(
                np.abs(X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]),
                axis=2
            )
            return distances

    # --------------------------------------------------------
    # API Methods
    # --------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Store training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        self.X_train_ = np.array(X, dtype=float)
        self.y_train_ = np.array(y)

        if len(self.X_train_) != len(self.y_train_):
            raise ValueError("X and y must have the same length.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        y_pred : np.ndarray
        """
        if self.X_train_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X = np.array(X, dtype=float)
        distances = self._compute_distances(X)

        # Indices of k nearest neighbors
        nn_idx = np.argsort(distances, axis=1)[:, :self.k]

        # Gather labels and vote
        nn_labels = self.y_train_[nn_idx]

        # Majority vote
        y_pred = np.array([
            np.bincount(row.astype(int)).argmax()
            for row in nn_labels
        ])

        return y_pred
