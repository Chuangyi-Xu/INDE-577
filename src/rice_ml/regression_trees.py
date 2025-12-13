"""
rice_ml.regression_trees

A simple CART-style Regression Tree implementation (MSE criterion).

API:
    - RegressionTreeRegressor.fit(X, y)
    - RegressionTreeRegressor.predict(X)

Notes:
    - This is a teaching / from-scratch implementation.
    - Splits are chosen to maximize variance reduction (equivalently minimize SSE/MSE).
    - Works best with numeric features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np


def _check_X_y(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D array-like, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array-like, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples, got {X.shape[0]} and {y.shape[0]}")
    if X.shape[0] == 0:
        raise ValueError("X has 0 samples")

    # NaN handling (simple: disallow)
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("NaNs detected in X or y. Please impute or drop missing values.")

    return X, y


def _check_X(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array-like, got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("X has 0 samples")
    if np.isnan(X).any():
        raise ValueError("NaNs detected in X. Please impute or drop missing values.")
    return X


def _mse(y: np.ndarray) -> float:
    # mean squared error around the mean == variance (up to constant); here use SSE/n for stability
    if y.size == 0:
        return 0.0
    mu = float(np.mean(y))
    return float(np.mean((y - mu) ** 2))


@dataclass
class _Node:
    is_leaf: bool
    value: float  # prediction at node (mean of y)

    feature_index: Optional[int] = None
    threshold: Optional[float] = None

    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    n_samples: int = 0
    impurity: float = 0.0  # MSE at node


class RegressionTreeRegressor:
    """
    CART-style regression tree using MSE (variance reduction).

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. None means unlimited (until other stopping rules trigger).
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : None, "sqrt", "log2", int, or float, default=None
        Number of features to consider when looking for best split.
        - None: use all features
        - "sqrt": sqrt(n_features)
        - "log2": log2(n_features)
        - int: exactly that many features
        - float in (0,1]: fraction of features
    random_state : int or None, default=None
        Random seed used when sampling features (max_features).
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, int, float]] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Optional[_Node] = None
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None

        self._rng = np.random.default_rng(random_state)

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, X, y):
        X, y = _check_X_y(X, y)

        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")

        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)

        self.root_ = self._build_tree(X, y, depth=0)

        # normalize importances
        total = float(np.sum(self.feature_importances_))
        if total > 0:
            self.feature_importances_ /= total

        return self

    def predict(self, X) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("This RegressionTreeRegressor instance is not fitted yet. Call fit() first.")
        X = _check_X(X)
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")

        preds = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            preds[i] = self._predict_one(X[i], self.root_)
        return preds

    # -------------------------
    # Internal methods
    # -------------------------
    def _predict_one(self, x: np.ndarray, node: _Node) -> float:
        while not node.is_leaf:
            # safety
            if node.feature_index is None or node.threshold is None:
                break
            if x[node.feature_index] <= node.threshold:
                node = node.left  # type: ignore
            else:
                node = node.right  # type: ignore
        return float(node.value)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples, n_features = X.shape
        node_value = float(np.mean(y))
        node_impurity = _mse(y)

        # stopping rules
        if n_samples < self.min_samples_split:
            return _Node(is_leaf=True, value=node_value, n_samples=n_samples, impurity=node_impurity)

        if self.max_depth is not None and depth >= self.max_depth:
            return _Node(is_leaf=True, value=node_value, n_samples=n_samples, impurity=node_impurity)

        # if y is constant-ish, stop
        if node_impurity <= 1e-15:
            return _Node(is_leaf=True, value=node_value, n_samples=n_samples, impurity=node_impurity)

        feat_idxs = self._sample_features(n_features)
        best = self._best_split(X, y, feat_idxs)

        if best is None:
            return _Node(is_leaf=True, value=node_value, n_samples=n_samples, impurity=node_impurity)

        best_feature, best_threshold, best_gain, left_mask = best
        if best_gain <= 0:
            return _Node(is_leaf=True, value=node_value, n_samples=n_samples, impurity=node_impurity)

        # split
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        # enforce min_samples_leaf
        if y_left.size < self.min_samples_leaf or y_right.size < self.min_samples_leaf:
            return _Node(is_leaf=True, value=node_value, n_samples=n_samples, impurity=node_impurity)

        # accumulate feature importance by impurity decrease weighted by samples
        self.feature_importances_[best_feature] += best_gain * n_samples

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return _Node(
            is_leaf=False,
            value=node_value,
            feature_index=int(best_feature),
            threshold=float(best_threshold),
            left=left_child,
            right=right_child,
            n_samples=n_samples,
            impurity=node_impurity,
        )

    def _sample_features(self, n_features: int) -> np.ndarray:
        mf = self.max_features
        if mf is None:
            k = n_features
        elif isinstance(mf, str):
            key = mf.lower()
            if key == "sqrt":
                k = int(np.sqrt(n_features))
            elif key == "log2":
                k = int(np.log2(n_features)) if n_features > 1 else 1
            else:
                raise ValueError('max_features must be None, "sqrt", "log2", int, or float')
            k = max(1, k)
        elif isinstance(mf, int):
            if mf <= 0 or mf > n_features:
                raise ValueError("int max_features must be in [1, n_features]")
            k = mf
        elif isinstance(mf, float):
            if mf <= 0.0 or mf > 1.0:
                raise ValueError("float max_features must be in (0, 1]")
            k = max(1, int(np.ceil(mf * n_features)))
        else:
            raise ValueError('max_features must be None, "sqrt", "log2", int, or float')

        if k == n_features:
            return np.arange(n_features)
        return self._rng.choice(n_features, size=k, replace=False)

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feat_idxs: np.ndarray,
    ) -> Optional[Tuple[int, float, float, np.ndarray]]:
        """
        Find the best (feature, threshold) split among feat_idxs using MSE reduction.

        Returns
        -------
        (best_feature, best_threshold, best_gain, left_mask) or None
        """
        n = y.size
        parent_impurity = _mse(y)

        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        best_left_mask = None

        for f in feat_idxs:
            x = X[:, f]

            # sort by feature values
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]

            # candidate thresholds: midpoints between unique x values
            # We also need fast impurity computation: use prefix sums
            # SSE = sum(y^2) - (sum(y)^2)/n
            y_prefix = np.cumsum(y_sorted)
            y2_prefix = np.cumsum(y_sorted ** 2)

            # valid split positions require distinct feature values and min leaf sizes
            # consider split between i and i+1 (left includes up to i)
            for i in range(self.min_samples_leaf - 1, n - self.min_samples_leaf):
                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                n_left = i + 1
                n_right = n - n_left

                sum_left = y_prefix[i]
                sum2_left = y2_prefix[i]
                sum_right = y_prefix[-1] - sum_left
                sum2_right = y2_prefix[-1] - sum2_left

                # mse = sse / n
                sse_left = sum2_left - (sum_left ** 2) / n_left
                sse_right = sum2_right - (sum_right ** 2) / n_right

                mse_left = sse_left / n_left
                mse_right = sse_right / n_right

                weighted_child_impurity = (n_left / n) * mse_left + (n_right / n) * mse_right
                gain = parent_impurity - weighted_child_impurity

                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(f)
                    best_threshold = float((x_sorted[i] + x_sorted[i + 1]) / 2.0)

                    # build mask in original order (not sorted order)
                    best_left_mask = X[:, f] <= best_threshold

        if best_feature is None or best_threshold is None or best_left_mask is None:
            return None

        return best_feature, best_threshold, best_gain, best_left_mask
