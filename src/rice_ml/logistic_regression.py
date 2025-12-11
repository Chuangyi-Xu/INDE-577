"""
Logistic Regression classifier implemented from scratch using NumPy.

This implementation follows a simple, scikit-learn–style API:

    from rice_ml.logistic_regression import LogisticRegression

    clf = LogisticRegression(learning_rate=0.1, n_epochs=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression classifier using batch gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent updates.

    n_epochs : int, default=1000
        Number of passes over the training dataset.

    fit_intercept : bool, default=True
        Whether to include an intercept (bias) term in the model.

    random_state : int or None, default=None
        Seed for NumPy's random number generator (used for weight initialization).

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_features,)
        Learned weight vector.

    bias_ : float
        Learned intercept term (0.0 if fit_intercept=False).

    classes_ : np.ndarray of shape (2,)
        Sorted unique class labels seen during `fit`.

    losses_ : list of float
        Binary cross-entropy loss values for each training epoch.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        # Attributes set during fit
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.classes_: Optional[np.ndarray] = None
        self.losses_: list[float] = []

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        if self.weights_ is None or self.classes_ is None:
            raise RuntimeError("This LogisticRegression instance is not fitted yet.")

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        # Clip to avoid overflow and log(0)
        z = np.clip(z, -250.0, 250.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """Compute linear combination w^T x + b."""
        return X @ self.weights_ + (self.bias_ if self.fit_intercept else 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit the logistic regression model on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Target class labels. Must contain exactly two distinct values
            (e.g., {0, 1} or {-1, 1}).

        Returns
        -------
        self : LogisticRegression
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        # Determine binary classes and map to {0, 1}
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(
                f"LogisticRegression supports binary classification only, "
                f"but got {classes.size} classes."
            )

        self.classes_ = classes
        # Use the second class as the "positive" class (mapped to 1)
        y_binary = (y == classes[1]).astype(float)

        n_samples, n_features = X.shape

        # Initialize weights
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0

        self.losses_ = []

        # Gradient descent loop
        for _ in range(self.n_epochs):
            # Linear model and predictions
            linear_output = self._net_input(X)
            y_pred = self._sigmoid(linear_output)

            # Binary cross-entropy loss
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
            loss = -np.mean(
                y_binary * np.log(y_pred_clipped)
                + (1.0 - y_binary) * np.log(1.0 - y_pred_clipped)
            )
            self.losses_.append(loss)

            # Gradients (average over samples)
            error = y_pred - y_binary  # shape (n_samples,)
            grad_w = X.T @ error / n_samples  # shape (n_features,)
            grad_b = float(np.sum(error) / n_samples)

            # Parameter update
            self.weights_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.bias_ -= self.learning_rate * grad_b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the raw decision scores (w^T x + b).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Linear decision scores.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        return self._net_input(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            Predicted probabilities for each class in `self.classes_`.
            Column 0 corresponds to `self.classes_[0]`,
            column 1 corresponds to `self.classes_[1]`.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        linear_output = self._net_input(X)
        p1 = self._sigmoid(linear_output)  # P(y = classes_[1])
        p0 = 1.0 - p1

        return np.vstack([p0, p1]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels from `self.classes_`.
        """
        proba = self.predict_proba(X)
        # Class index with max probability (0 or 1)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float
            Mean accuracy of predictions.
        """
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
