import numpy as np
import pytest

from rice_ml.pca import PCA


def test_pca_fit_shapes():
    """
    Test whether PCA learns correct shapes of attributes.
    """
    np.random.seed(0)
    X = np.random.randn(100, 5)

    pca = PCA(n_components=2)
    pca.fit(X)

    assert pca.components_.shape == (2, 5)
    assert pca.explained_variance_.shape == (2,)
    assert pca.explained_variance_ratio_.shape == (2,)
    assert pca.mean_.shape == (5,)


def test_pca_transform_shape():
    """
    Test output shape after PCA transform.
    """
    X = np.random.randn(50, 4)

    pca = PCA(n_components=3)
    pca.fit(X)
    X_transformed = pca.transform(X)

    assert X_transformed.shape == (50, 3)


def test_pca_fit_transform_equivalence():
    """
    Test fit_transform equals fit + transform.
    """
    X = np.random.randn(30, 6)

    pca1 = PCA(n_components=2)
    Xt1 = pca1.fit_transform(X)

    pca2 = PCA(n_components=2)
    pca2.fit(X)
    Xt2 = pca2.transform(X)

    assert np.allclose(Xt1, Xt2)


def test_pca_variance_ratio_sum():
    """
    Explained variance ratio should sum to <= 1.
    """
    X = np.random.randn(100, 10)

    pca = PCA(n_components=5)
    pca.fit(X)

    total_ratio = np.sum(pca.explained_variance_ratio_)
    assert total_ratio <= 1.0 + 1e-8
    assert total_ratio > 0.0


def test_pca_not_fitted_error():
    """
    Transform before fit should raise an error.
    """
    X = np.random.randn(10, 3)
    pca = PCA(n_components=2)

    with pytest.raises(ValueError):
        pca.transform(X)
