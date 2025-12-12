import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA)

    A simple NumPy-based implementation of PCA using eigen-decomposition
    of the covariance matrix.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components to keep.
        If None, all components are kept.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space.

    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each principal component.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each component.

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

        # attributes after fitting
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Fit the PCA model with X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues & eigenvectors (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # 5. Select number of components
        if self.n_components is None:
            n_components = n_features
        else:
            n_components = self.n_components

        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]

        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
        )

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("PCA has not been fitted yet.")

        X = np.asarray(X)
        X_centered = X - self.mean_

        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        """
        Fit PCA model and apply dimensionality reduction to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
