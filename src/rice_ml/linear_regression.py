# src/rice_ml/linear_regression.py

import numpy as np


class LinearRegression:
    """
    Linear Regression implemented with Gradient Descent.
    Matches scikit-learn style: fit(), predict(), score().

    Parameters
    ----------
    lr : float
        Learning rate for gradient descent.
    num_iter : int
        Maximum number of iterations.
    fit_intercept : bool
        Whether to add a bias term.
    tol : float
        Convergence tolerance.

    Attributes
    ----------
    weights_ : ndarray
        Learned coefficients.
    bias_ : float
        Intercept term (if fit_intercept=True).
    history_ : list
        Loss values per iteration.
    """

    def __init__(self, lr=0.01, num_iter=1000, fit_intercept=True, tol=1e-6):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.tol = tol

        # learned parameters
        self.weights_ = None
        self.bias_ = None

        # loss history (MSE)
        self.history_ = []

    # ------------------------------------------------------------
    # Utility: add bias column if needed (not used here—bias is scalar)
    # ------------------------------------------------------------
    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.c_[np.ones(X.shape[0]), X]

    # ------------------------------------------------------------
    # MSE loss
    # ------------------------------------------------------------
    def _mse_loss(self, y, y_pred):
        return np.mean((y_pred - y) ** 2)

    # ------------------------------------------------------------
    # Fit model using gradient descent
    # ------------------------------------------------------------
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        # initialize parameters
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        for i in range(self.num_iter):

            # prediction
            y_pred = np.dot(X, self.weights_) + self.bias_

            # compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights_ -= self.lr * dw
            self.bias_ -= self.lr * db

            # compute loss for history
            mse = self._mse_loss(y, y_pred)
            self.history_.append(mse)

            # check convergence
            if i > 0 and abs(self.history_[-2] - mse) < self.tol:
                break

        return self

    # ------------------------------------------------------------
    # Predict continuous values
    # ------------------------------------------------------------
    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights_) + self.bias_

    # ------------------------------------------------------------
    # R² score
    # ------------------------------------------------------------
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
