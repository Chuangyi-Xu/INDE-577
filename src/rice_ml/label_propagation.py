import numpy as np


class LabelPropagation:
    """
    Label Propagation Algorithm for semi-supervised learning.

    This implementation follows the classic graph-based label propagation
    approach (Zhu & Ghahramani, 2002).

    Parameters
    ----------
    gamma : float, default=1.0
        Kernel coefficient for the RBF (Gaussian) similarity function.

    max_iter : int, default=1000
        Maximum number of label propagation iterations.

    tol : float, default=1e-3
        Convergence tolerance. Iterations stop when label changes are below this threshold.
    """

    def __init__(self, gamma=1.0, max_iter=1000, tol=1e-3):
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

        self.X_ = None
        self.y_ = None
        self.label_distributions_ = None
        self.classes_ = None

    def _rbf_kernel(self, X):
        """
        Compute RBF (Gaussian) similarity matrix.
        """
        sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * sq_dists)

    def fit(self, X, y):
        """
        Fit the label propagation model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Labels. Unlabeled points should be marked as -1.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.X_ = X
        self.y_ = y

        # Identify labeled and unlabeled samples
        labeled_mask = y != -1

        # Unique class labels
        self.classes_ = np.unique(y[labeled_mask])
        n_classes = len(self.classes_)
        n_samples = X.shape[0]

        # Build similarity graph
        W = self._rbf_kernel(X)

        # Normalize similarity matrix (row-stochastic)
        W = W / W.sum(axis=1, keepdims=True)

        # Initialize label distributions
        F = np.zeros((n_samples, n_classes))

        for idx, label in enumerate(self.classes_):
            F[y == label, idx] = 1.0

        # Label propagation iterations
        for _ in range(self.max_iter):
            F_old = F.copy()

            F = W @ F

            # Clamp labeled points
            for idx, label in enumerate(self.classes_):
                F[y == label, :] = 0.0
                F[y == label, idx] = 1.0

            # Check convergence
            if np.linalg.norm(F - F_old) < self.tol:
                break

        self.label_distributions_ = F
        return self

    def predict(self, X=None):
        """
        Predict labels for the data.

        Parameters
        ----------
        X : ignored
            Included for API compatibility.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        if self.label_distributions_ is None:
            raise ValueError("Model has not been fitted yet.")

        class_indices = np.argmax(self.label_distributions_, axis=1)
        return self.classes_[class_indices]

    def fit_predict(self, X, y):
        """
        Fit the model and return predicted labels.
        """
        self.fit(X, y)
        return self.predict()
