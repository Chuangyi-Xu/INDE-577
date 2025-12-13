import numpy as np
from typing import Optional


class KMeans:
    """
    K-Means Clustering Algorithm (from scratch)

    Parameters
    ----------
    n_clusters : int
        Number of clusters (k).

    max_iter : int, default=300
        Maximum number of iterations.

    tol : float, default=1e-4
        Convergence tolerance based on centroid movement.

    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # attributes set after fitting
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Randomly initialize centroids by sampling data points.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.random.choice(
            X.shape[0], self.n_clusters, replace=False
        )
        return X[indices]

    def _assign_clusters(
        self, X: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """
        Assign each point to the nearest centroid.
        """
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - centroids[np.newaxis, :, :],
            axis=2,
        )
        return np.argmin(distances, axis=1)

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Update centroids as the mean of assigned points.
        """
        new_centroids = np.zeros(
            (self.n_clusters, X.shape[1])
        )

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if cluster_points.shape[0] == 0:
                # keep old centroid if cluster is empty
                new_centroids[k] = self.cluster_centers_[k]
            else:
                new_centroids[k] = cluster_points.mean(axis=0)

        return new_centroids

    def _compute_inertia(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        """
        Compute within-cluster sum of squared distances (WCSS).
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            inertia += np.sum(
                (cluster_points - centroids[k]) ** 2
            )
        return inertia

    def fit(self, X: np.ndarray):
        """
        Fit K-Means clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                "Number of samples must be >= n_clusters."
            )

        # initialize centroids
        self.cluster_centers_ = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X, self.cluster_centers_)
            new_centroids = self._update_centroids(X, labels)

            shift = np.linalg.norm(
                new_centroids - self.cluster_centers_
            )
            self.cluster_centers_ = new_centroids

            if shift < self.tol:
                break

        self.labels_ = self._assign_clusters(
            X, self.cluster_centers_
        )
        self.inertia_ = self._compute_inertia(
            X, self.labels_, self.cluster_centers_
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        if self.cluster_centers_ is None:
            raise RuntimeError(
                "KMeans instance is not fitted yet."
            )

        X = np.asarray(X)
        return self._assign_clusters(X, self.cluster_centers_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit K-Means and return cluster labels.
        """
        self.fit(X)
        return self.labels_
