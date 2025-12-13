import numpy as np
import pytest

from rice_ml.kmeans import KMeans


def test_kmeans_fit_basic():
    """
    Basic functionality test: fit should run and set attributes.
    """
    X = np.array([
        [1.0, 1.0],
        [1.1, 1.0],
        [0.9, 1.2],
        [5.0, 5.0],
        [5.1, 4.9],
        [4.9, 5.2],
    ])

    model = KMeans(n_clusters=2, random_state=0)
    model.fit(X)

    assert model.cluster_centers_ is not None
    assert model.labels_ is not None
    assert model.inertia_ is not None

    assert model.cluster_centers_.shape == (2, 2)
    assert model.labels_.shape == (X.shape[0],)


def test_kmeans_fit_predict_consistency():
    """
    fit_predict should return the same labels as fit + labels_.
    """
    X = np.random.RandomState(42).rand(20, 3)

    model = KMeans(n_clusters=3, random_state=42)
    labels_fit_predict = model.fit_predict(X)

    assert labels_fit_predict.shape == (20,)
    assert np.array_equal(labels_fit_predict, model.labels_)


def test_kmeans_predict_after_fit():
    """
    predict should assign clusters after model is fitted.
    """
    X_train = np.array([
        [0.0, 0.0],
        [0.1, 0.2],
        [9.8, 10.0],
        [10.1, 9.9],
    ])

    X_test = np.array([
        [0.05, 0.1],
        [10.0, 10.0],
    ])

    model = KMeans(n_clusters=2, random_state=1)
    model.fit(X_train)
    labels = model.predict(X_test)

    assert labels.shape == (2,)
    assert np.issubdtype(labels.dtype, np.integer)


def test_kmeans_reproducibility_with_random_state():
    """
    Using the same random_state should produce identical results.
    """
    X = np.random.rand(50, 2)

    model1 = KMeans(n_clusters=3, random_state=123)
    model2 = KMeans(n_clusters=3, random_state=123)

    model1.fit(X)
    model2.fit(X)

    assert np.allclose(
        model1.cluster_centers_,
        model2.cluster_centers_
    )
    assert np.array_equal(
        model1.labels_,
        model2.labels_
    )


def test_kmeans_predict_without_fit_raises_error():
    """
    Calling predict before fit should raise an error.
    """
    X = np.random.rand(10, 2)
    model = KMeans(n_clusters=2)

    with pytest.raises(RuntimeError):
        model.predict(X)


def test_kmeans_invalid_input_dimension():
    """
    Input X must be a 2D array.
    """
    X = np.array([1.0, 2.0, 3.0])
    model = KMeans(n_clusters=2)

    with pytest.raises(ValueError):
        model.fit(X)


def test_kmeans_more_clusters_than_samples():
    """
    n_clusters cannot exceed number of samples.
    """
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])

    model = KMeans(n_clusters=3)

    with pytest.raises(ValueError):
        model.fit(X)
