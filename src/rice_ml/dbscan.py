# src/rice_ml/dbscan.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np


def _euclidean_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between rows of X and rows of Y.
    Returns shape (n_samples_X, n_samples_Y).
    """
    # (x - y)^2 = x^2 + y^2 - 2xy
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X2 = np.sum(X * X, axis=1, keepdims=True)          # (nX, 1)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True).T        # (1, nY)
    D2 = X2 + Y2 - 2.0 * (X @ Y.T)
    # numerical safety
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2)


MetricType = Union[Literal["euclidean"], Callable[[np.ndarray, np.ndarray], np.ndarray]]


@dataclass
class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered a core point.
        This includes the point itself.
    metric : {"euclidean"} or callable, default="euclidean"
        Distance metric. If callable, should take (X, Y) and return pairwise distances
        with shape (n_samples_X, n_samples_Y).
    """

    eps: float = 0.5
    min_samples: int = 5
    metric: MetricType = "euclidean"

    # learned attributes
    labels_: Optional[np.ndarray] = None
    core_sample_indices_: Optional[np.ndarray] = None
    components_: Optional[np.ndarray] = None

    def _validate_params(self) -> None:
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.min_samples <= 0:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")

        if not (self.metric == "euclidean" or callable(self.metric)):
            raise ValueError("metric must be 'euclidean' or a callable (X, Y) -> distances")

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return _euclidean_distances(X, X)
        # callable metric
        return np.asarray(self.metric(X, X), dtype=float)

    def _region_query(self, D: np.ndarray, i: int) -> np.ndarray:
        """
        Return indices of neighbors of point i (including itself) within eps.
        """
        return np.where(D[i] <= self.eps)[0]

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Fit DBSCAN clustering from features X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self._validate_params()

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        n = X.shape[0]
        if n == 0:
            # empty input
            self.labels_ = np.array([], dtype=int)
            self.core_sample_indices_ = np.array([], dtype=int)
            self.components_ = np.empty((0, X.shape[1]), dtype=float)
            return self

        # Compute pairwise distances (brute force)
        D = self._pairwise_distances(X)

        labels = np.full(n, -1, dtype=int)  # -1 means noise
        visited = np.zeros(n, dtype=bool)

        # Precompute core points
        neighbor_counts = np.sum(D <= self.eps, axis=1)
        is_core = neighbor_counts >= self.min_samples
        core_indices = np.where(is_core)[0]

        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._region_query(D, i)

            if neighbors.size == 0 or (neighbors.size < self.min_samples):
                # not a core point -> remains noise (unless later added to a cluster as border)
                continue

            # start a new cluster
            labels[i] = cluster_id

            # Expand cluster
            # We maintain a queue of points to process
            queue = list(neighbors.tolist())

            # Important: DBSCAN expansion rule
            # - If a neighbor is a core point, add its neighbors to the queue
            # - If a neighbor is unassigned (noise or not yet labeled), assign it to the cluster
            qpos = 0
            while qpos < len(queue):
                j = queue[qpos]
                qpos += 1

                if not visited[j]:
                    visited[j] = True
                    j_neighbors = self._region_query(D, j)

                    if j_neighbors.size >= self.min_samples:
                        # core point => add its neighbors to queue
                        # avoid too many duplicates with a simple membership check using a boolean mask
                        # (n is typically not huge in course projects; keep simple & robust)
                        for t in j_neighbors.tolist():
                            if t not in queue:
                                queue.append(t)

                # Assign to cluster if not yet assigned
                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = core_indices
        self.components_ = X[core_indices].copy()
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN and return cluster labels.
        """
        self.fit(X)
        return self.labels_.copy()
