import numpy as np
import pytest

from rice_ml.label_propagation import LabelPropagation


def test_label_propagation_fit_runs():
    """
    fit should run without error on a simple dataset.
    """
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 1.1],
    ])

    # Two labeled points, two unlabeled
    y = np.array([0, -1, 1, -1])

    model = LabelPropagation(gamma=1.0, max_iter=100)
    model.fit(X, y)

    assert model.label_distributions_ is not None
    assert model.classes_ is not None


def test_label_propagation_predict_shape():
    """
    predict should return an array of correct shape.
    """
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 1.1],
    ])
    y = np.array([0, -1, 1, -1])

    model = LabelPropagation(gamma=1.0, max_iter=100)
    model.fit(X, y)

    y_pred = model.predict()

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (X.shape[0],)


def test_label_propagation_respects_labeled_points():
    """
    Labeled points should keep their original labels after propagation.
    """
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 1.1],
    ])
    y = np.array([0, -1, 1, -1])

    model = LabelPropagation(gamma=1.0, max_iter=100)
    y_pred = model.fit_predict(X, y)

    # Check labeled points unchanged
    assert y_pred[0] == 0
    assert y_pred[2] == 1


def test_label_propagation_clusters_correctly():
    """
    Unlabeled points near a class should receive the correct label.
    """
    X = np.array([
        [0.0, 0.0],   # class 0
        [0.2, 0.1],   # unlabeled, near class 0
        [5.0, 5.0],   # class 1
        [5.1, 5.2],   # unlabeled, near class 1
    ])
    y = np.array([0, -1, 1, -1])

    model = LabelPropagation(gamma=0.5, max_iter=200)
    y_pred = model.fit_predict(X, y)

    assert y_pred[1] == 0
    assert y_pred[3] == 1


def test_predict_before_fit_raises_error():
    """
    predict before fit should raise an error.
    """
    model = LabelPropagation()

    with pytest.raises(ValueError):
        model.predict()
