"""
rice_ml.decision_trees
======================

A lightweight, NumPy-only implementation of CART-style Decision Trees for:
- Classification (Gini / Entropy)
- Regression (MSE / MAE)

Design goals:
- Scikit-learn-like API: fit / predict / predict_proba
- Deterministic behavior with random_state
- Reasonable defaults + clean, readable implementation for educational use

Limitations:
- Only supports numeric features (float/int)
- No native missing-value handling (NaN will raise)
- No cost-complexity pruning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _check_X_y(X: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D array-like. Got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array-like. Got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples. Got {X.shape[0]} and {y.shape[0]}.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf. This implementation does not support missing values.")
    if not np.isfinite(y).all() and y.dtype.kind in {"f"}:
        raise ValueError("y contains NaN or inf.")

    return X.astype(float, copy=False), y


def _check_X(X: Any) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array-like. Got shape {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf. This implementation does not support missing values.")
    return X.astype(float, copy=False)


def _rng(random_state: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(None if random_state is None else int(random_state))


def _weighted_counts(y_int: np.ndarray, sample_weight: Optional[np.ndarray], n_classes: int) -> np.ndarray:
    if sample_weight is None:
        return np.bincount(y_int, minlength=n_classes).astype(float)
    counts = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        counts[c] = sample_weight[y_int == c].sum()
    return counts


def _safe_log2(p: np.ndarray) -> np.ndarray:
    # avoid log(0)
    p = np.clip(p, 1e-15, 1.0)
    return np.log2(p)


# ---------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------
@dataclass
class _Node:
    is_leaf: bool
    # for internal nodes
    feature_index: int = -1
    threshold: float = 0.0
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    # for leaves
    value: Optional[np.ndarray] = None  # classifier: class distribution; regressor: scalar in array([v])
    n_samples: int = 0
    impurity: float = 0.0


# ---------------------------------------------------------------------
# Base Tree
# ---------------------------------------------------------------------
class _BaseDecisionTree:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = float(min_impurity_decrease)

        self.n_features_in_: Optional[int] = None
        self.root_: Optional[_Node] = None
        self.feature_importances_: Optional[np.ndarray] = None

        self._rng = _rng(random_state)

    def _validate_common_hparams(self) -> None:
        if self.max_depth is not None and int(self.max_depth) < 1:
            raise ValueError("max_depth must be None or >= 1.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        if self.min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be >= 0.")

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            mf = mf.lower()
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError("max_features string must be one of {None, 'sqrt', 'log2'}.")
        if isinstance(mf, float):
            if not (0.0 < mf <= 1.0):
                raise ValueError("max_features as float must be in (0, 1].")
            return max(1, int(np.ceil(mf * n_features)))
        if isinstance(mf, int):
            if mf < 1 or mf > n_features:
                raise ValueError("max_features as int must be in [1, n_features].")
            return mf
        raise ValueError("max_features must be None, int, float, or {'sqrt','log2'}.")

    # --- methods subclasses must implement ---
    def _impurity(self, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> float:
        raise NotImplementedError

    def _leaf_value(self, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> Tuple[Optional[int], Optional[float], float]:
        raise NotImplementedError

    # --- generic build ---
    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        depth: int,
        importances_accum: np.ndarray,
    ) -> _Node:
        n_samples, n_features = X.shape
        node_impurity = self._impurity(y, sample_weight)

        # Stopping conditions
        if n_samples < self.min_samples_split:
            return _Node(is_leaf=True, value=self._leaf_value(y, sample_weight), n_samples=n_samples, impurity=node_impurity)

        if self.max_depth is not None and depth >= int(self.max_depth):
            return _Node(is_leaf=True, value=self._leaf_value(y, sample_weight), n_samples=n_samples, impurity=node_impurity)

        if n_samples <= 2 * self.min_samples_leaf:
            return _Node(is_leaf=True, value=self._leaf_value(y, sample_weight), n_samples=n_samples, impurity=node_impurity)

        # Find best split
        feat_idx, thr, impurity_decrease = self._best_split(X, y, sample_weight)

        if feat_idx is None or thr is None:
            return _Node(is_leaf=True, value=self._leaf_value(y, sample_weight), n_samples=n_samples, impurity=node_impurity)

        if impurity_decrease < self.min_impurity_decrease:
            return _Node(is_leaf=True, value=self._leaf_value(y, sample_weight), n_samples=n_samples, impurity=node_impurity)

        # Partition
        left_mask = X[:, feat_idx] <= thr
        right_mask = ~left_mask

        # Safety (shouldn't happen if we respect min_samples_leaf)
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return _Node(is_leaf=True, value=self._leaf_value(y, sample_weight), n_samples=n_samples, impurity=node_impurity)

        # Track impurity decrease for feature importance
        importances_accum[feat_idx] += impurity_decrease

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        w_left = sample_weight[left_mask] if sample_weight is not None else None
        w_right = sample_weight[right_mask] if sample_weight is not None else None

        left_node = self._build(X_left, y_left, w_left, depth + 1, importances_accum)
        right_node = self._build(X_right, y_right, w_right, depth + 1, importances_accum)

        return _Node(
            is_leaf=False,
            feature_index=int(feat_idx),
            threshold=float(thr),
            left=left_node,
            right=right_node,
            value=None,
            n_samples=n_samples,
            impurity=node_impurity,
        )

    def _predict_row(self, x: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        node = self.root_
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
        return node.value  # type: ignore[return-value]


# ---------------------------------------------------------------------
# DecisionTreeClassifier
# ---------------------------------------------------------------------
class DecisionTreeClassifier(_BaseDecisionTree):
    """
    CART-style Decision Tree Classifier.

    Parameters
    ----------
    criterion : {"gini", "entropy"}
    max_depth : int or None
    min_samples_split : int
    min_samples_leaf : int
    max_features : None, int, float, {"sqrt", "log2"}
    random_state : int or None
    min_impurity_decrease : float
    class_weight : None or dict or "balanced"
        - None: no weighting
        - dict: {class_label: weight}
        - "balanced": weights inversely proportional to class frequencies
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Optional[Union[Dict[Any, float], str]] = None,
    ) -> None:
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.criterion = str(criterion).lower()
        self.class_weight = class_weight

        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None

    def _validate(self) -> None:
        self._validate_common_hparams()
        if self.criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if self.class_weight is not None and not (isinstance(self.class_weight, dict) or self.class_weight == "balanced"):
            raise ValueError("class_weight must be None, a dict, or 'balanced'.")

    def _make_sample_weight(self, y: np.ndarray) -> Optional[np.ndarray]:
        if self.class_weight is None:
            return None

        classes, y_int = np.unique(y, return_inverse=True)
        counts = np.bincount(y_int)

        if self.class_weight == "balanced":
            # sklearn-like: n_samples / (n_classes * count_c)
            n_samples = len(y_int)
            weights_per_class = n_samples / (len(classes) * np.maximum(counts, 1))
            return weights_per_class[y_int].astype(float)

        # dict case
        w_map = {k: float(v) for k, v in self.class_weight.items()}  # type: ignore[union-attr]
        w = np.ones_like(y_int, dtype=float)
        for i, c in enumerate(classes):
            if c in w_map:
                w[y_int == i] = w_map[c]
        return w

    def _impurity(self, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> float:
        # y here is already integer-coded in _fit_internal
        y_int = y.astype(int, copy=False)
        n_classes = int(self.n_classes_)  # type: ignore[arg-type]
        counts = _weighted_counts(y_int, sample_weight, n_classes)
        total = counts.sum()
        if total <= 0:
            return 0.0
        p = counts / total
        if self.criterion == "gini":
            return float(1.0 - np.sum(p * p))
        # entropy
        return float(-np.sum(p * _safe_log2(p)))

    def _leaf_value(self, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> np.ndarray:
        y_int = y.astype(int, copy=False)
        n_classes = int(self.n_classes_)  # type: ignore[arg-type]
        counts = _weighted_counts(y_int, sample_weight, n_classes)
        total = counts.sum()
        if total <= 0:
            # fallback uniform
            return np.ones(n_classes, dtype=float) / n_classes
        return counts / total

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> Tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        max_feats = self._resolve_max_features(n_features)

        # choose subset of features (like RandomForest behavior)
        feat_candidates = np.arange(n_features)
        if max_feats < n_features:
            feat_candidates = self._rng.choice(feat_candidates, size=max_feats, replace=False)

        parent_impurity = self._impurity(y, sample_weight)
        best_feat: Optional[int] = None
        best_thr: Optional[float] = None
        best_gain: float = 0.0

        # total weight
        w_total = float(sample_weight.sum()) if sample_weight is not None else float(n_samples)

        for f in feat_candidates:
            x = X[:, f]

            # Sort by feature for efficient threshold scan
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            w_sorted = sample_weight[order] if sample_weight is not None else None

            # Candidate thresholds: midpoints between distinct values
            # If all equal -> no split
            if x_sorted[0] == x_sorted[-1]:
                continue

            # Precompute cumulative class counts on the left
            n_classes = int(self.n_classes_)  # type: ignore[arg-type]
            if w_sorted is None:
                left_counts = np.zeros(n_classes, dtype=float)
                total_counts = np.bincount(y_sorted.astype(int), minlength=n_classes).astype(float)
                left_w = 0.0
                total_w = float(n_samples)
            else:
                total_counts = _weighted_counts(y_sorted.astype(int), w_sorted, n_classes)
                left_counts = np.zeros(n_classes, dtype=float)
                left_w = 0.0
                total_w = float(w_sorted.sum())

            # iterate split point i where left=[0..i], right=[i+1..]
            for i in range(n_samples - 1):
                cls = int(y_sorted[i])
                wi = float(w_sorted[i]) if w_sorted is not None else 1.0
                left_counts[cls] += wi
                left_w += wi

                # enforce min_samples_leaf in terms of counts (not weights)
                # sample-count constraint: i+1 left, n_samples-i-1 right
                if (i + 1) < self.min_samples_leaf:
                    continue
                if (n_samples - (i + 1)) < self.min_samples_leaf:
                    break

                # skip if identical value -> no valid threshold between i and i+1
                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                right_counts = total_counts - left_counts
                right_w = total_w - left_w
                if left_w <= 0 or right_w <= 0:
                    continue

                # impurity(left)
                pL = left_counts / left_w
                if self.criterion == "gini":
                    impL = 1.0 - float(np.sum(pL * pL))
                else:
                    impL = -float(np.sum(pL * _safe_log2(pL)))

                # impurity(right)
                pR = right_counts / right_w
                if self.criterion == "gini":
                    impR = 1.0 - float(np.sum(pR * pR))
                else:
                    impR = -float(np.sum(pR * _safe_log2(pR)))

                # weighted child impurity
                child_imp = (left_w / total_w) * impL + (right_w / total_w) * impR
                gain = parent_impurity - child_imp

                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thr = float(0.5 * (x_sorted[i] + x_sorted[i + 1]))

        # impurity decrease scaled by node weight (like contribution)
        # This makes importances accumulate more meaningfully.
        impurity_decrease = best_gain * w_total
        if best_feat is None:
            return None, None, 0.0
        return best_feat, best_thr, float(impurity_decrease)

    def fit(self, X: Any, y: Any) -> "DecisionTreeClassifier":
        self._validate()
        X, y_raw = _check_X_y(X, y)

        # encode classes
        classes, y_int = np.unique(y_raw, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = int(len(classes))

        self.n_features_in_ = int(X.shape[1])

        # build sample_weight from class_weight (optional)
        sw = self._make_sample_weight(y_raw)

        importances_accum = np.zeros(self.n_features_in_, dtype=float)
        self.root_ = self._build(X, y_int.astype(int), sw, depth=0, importances_accum=importances_accum)

        total = importances_accum.sum()
        if total > 0:
            self.feature_importances_ = importances_accum / total
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = _check_X(X)
        if self.root_ is None or self.classes_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        probs = np.zeros((X.shape[0], int(self.n_classes_)), dtype=float)  # type: ignore[arg-type]
        for i in range(X.shape[0]):
            probs[i] = self._predict_row(X[i])
        return probs

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]  # type: ignore[index]


# ---------------------------------------------------------------------
# DecisionTreeRegressor
# ---------------------------------------------------------------------
class DecisionTreeRegressor(_BaseDecisionTree):
    """
    CART-style Decision Tree Regressor.

    Parameters
    ----------
    criterion : {"mse", "mae"}
    max_depth : int or None
    min_samples_split : int
    min_samples_leaf : int
    max_features : None, int, float, {"sqrt", "log2"}
    random_state : int or None
    min_impurity_decrease : float
    """

    def __init__(
        self,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.criterion = str(criterion).lower()

    def _validate(self) -> None:
        self._validate_common_hparams()
        if self.criterion not in {"mse", "mae"}:
            raise ValueError("criterion must be 'mse' or 'mae'.")

    def _impurity(self, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> float:
        y = y.astype(float, copy=False)
        if y.size == 0:
            return 0.0
        if sample_weight is None:
            if self.criterion == "mse":
                mu = float(np.mean(y))
                return float(np.mean((y - mu) ** 2))
            # mae
            med = float(np.median(y))
            return float(np.mean(np.abs(y - med)))

        w = sample_weight.astype(float, copy=False)
        w_sum = float(w.sum())
        if w_sum <= 0:
            return 0.0

        if self.criterion == "mse":
            mu = float(np.sum(w * y) / w_sum)
            return float(np.sum(w * (y - mu) ** 2) / w_sum)

        # weighted median for MAE (approx via sort)
        order = np.argsort(y)
        y_sorted = y[order]
        w_sorted = w[order]
        cum = np.cumsum(w_sorted)
        med_idx = int(np.searchsorted(cum, 0.5 * w_sum, side="left"))
        med = float(y_sorted[min(med_idx, len(y_sorted) - 1)])
        return float(np.sum(w * np.abs(y - med)) / w_sum)

    def _leaf_value(self, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> np.ndarray:
        y = y.astype(float, copy=False)
        if sample_weight is None:
            return np.array([float(np.mean(y))], dtype=float)

        w = sample_weight.astype(float, copy=False)
        w_sum = float(w.sum())
        if w_sum <= 0:
            return np.array([float(np.mean(y))], dtype=float)
        return np.array([float(np.sum(w * y) / w_sum)], dtype=float)

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> Tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        max_feats = self._resolve_max_features(n_features)

        feat_candidates = np.arange(n_features)
        if max_feats < n_features:
            feat_candidates = self._rng.choice(feat_candidates, size=max_feats, replace=False)

        parent_impurity = self._impurity(y, sample_weight)
        best_feat: Optional[int] = None
        best_thr: Optional[float] = None
        best_gain: float = 0.0

        w_total = float(sample_weight.sum()) if sample_weight is not None else float(n_samples)

        for f in feat_candidates:
            x = X[:, f]
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            w_sorted = sample_weight[order] if sample_weight is not None else None

            if x_sorted[0] == x_sorted[-1]:
                continue

            # scan split points
            for i in range(n_samples - 1):
                # count constraints
                if (i + 1) < self.min_samples_leaf:
                    continue
                if (n_samples - (i + 1)) < self.min_samples_leaf:
                    break
                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                thr = float(0.5 * (x_sorted[i] + x_sorted[i + 1]))

                yL = y_sorted[: i + 1]
                yR = y_sorted[i + 1 :]

                if w_sorted is None:
                    impL = self._impurity(yL, None)
                    impR = self._impurity(yR, None)
                    child_imp = ((i + 1) / n_samples) * impL + ((n_samples - (i + 1)) / n_samples) * impR
                else:
                    wL = w_sorted[: i + 1]
                    wR = w_sorted[i + 1 :]
                    wL_sum = float(wL.sum())
                    wR_sum = float(wR.sum())
                    if wL_sum <= 0 or wR_sum <= 0:
                        continue
                    impL = self._impurity(yL, wL)
                    impR = self._impurity(yR, wR)
                    child_imp = (wL_sum / (wL_sum + wR_sum)) * impL + (wR_sum / (wL_sum + wR_sum)) * impR

                gain = parent_impurity - child_imp
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thr = thr

        impurity_decrease = best_gain * w_total
        if best_feat is None:
            return None, None, 0.0
        return best_feat, best_thr, float(impurity_decrease)

    def fit(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> "DecisionTreeRegressor":
        self._validate()
        X, y = _check_X_y(X, y)
        y = y.astype(float, copy=False)

        sw: Optional[np.ndarray]
        if sample_weight is None:
            sw = None
        else:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.ndim != 1 or sw.shape[0] != X.shape[0]:
                raise ValueError("sample_weight must be 1D array-like with same length as X.")
            if (sw < 0).any():
                raise ValueError("sample_weight must be non-negative.")
            if not np.isfinite(sw).all():
                raise ValueError("sample_weight contains NaN or inf.")

        self.n_features_in_ = int(X.shape[1])

        importances_accum = np.zeros(self.n_features_in_, dtype=float)
        self.root_ = self._build(X, y, sw, depth=0, importances_accum=importances_accum)

        total = importances_accum.sum()
        if total > 0:
            self.feature_importances_ = importances_accum / total
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)

        return self

    def predict(self, X: Any) -> np.ndarray:
        X = _check_X(X)
        if self.root_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        y_pred = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            y_pred[i] = float(self._predict_row(X[i])[0])
        return y_pred
