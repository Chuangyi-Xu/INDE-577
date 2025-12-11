import numpy as np
import pytest

from rice_ml.logistic_regression import LogisticRegression


def test_logistic_regression_fit_dimensions():
    """Check weight dimension, bias scalar, and classes_ shape."""
    X = np.array([[0, 1], [1, 1], [2, 1], [3, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = LogisticRegression(learning_rate=0.1, n_epochs=500, random_state=0)
    clf.fit(X, y)

    # weights shape
    assert clf.weights_.shape == (2,)
    # bias exists when fit_intercept=True
    assert isinstance(clf.bias_, float)
    # classes_ contains exactly two sorted labels
    assert np.array_equal(clf.classes_, np.array([0, 1]))


def test_logistic_regression_predict_binary_linearly_separable():
    """Test logistic regression on an easily separable dataset."""
    # Simple dataset: label = 1 when x1 > 1.5
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = LogisticRegression(learning_rate=0.5, n_epochs=2000, random_state=0)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert np.array_equal(preds, y)

    # score should be perfect
    assert clf.score(X, y) == 1.0


def test_logistic_regression_predict_proba_sum_to_one():
    """Check predict_proba returns valid probability distribution."""
    X = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0, 1, 1])

    clf = LogisticRegression(learning_rate=0.1, n_epochs=500, random_state=0)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    # shape should be (n_samples, 2)
    assert proba.shape == (3, 2)
    # rows should sum to 1
    np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(3))


def test_logistic_regression_loss_decreasing():
    """Ensure that training loss decreases over epochs."""
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = LogisticRegression(learning_rate=0.1, n_epochs=800, random_state=0)
    clf.fit(X, y)

    losses = clf.losses_
    assert len(losses) == 800
    # final loss should be smaller than initial loss
    assert losses[-1] < losses[0]


def test_logistic_regression_predict_after_fit():
    """Check predict raises error before fit, works after fit."""
    X = np.array([[0], [1]], dtype=float)
    y = np.array([0, 1])

    clf = LogisticRegression()

    # not fitted yet
    with pytest.raises(RuntimeError):
        clf.predict(X)

    # now fit
    clf.fit(X, y)
    preds = clf.predict(X)

    assert preds.shape == (2,)
    assert set(preds).issubset({0, 1})
