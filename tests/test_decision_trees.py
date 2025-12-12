import numpy as np
import pytest

from rice_ml.decision_trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


# =========================================================
# Classification Tests
# =========================================================

def test_decision_tree_classifier_fit_predict_simple():
    """
    Simple linearly separable dataset
    """
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=2,
        random_state=42,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert y_pred.shape == y.shape
    assert np.array_equal(y_pred, y)


def test_decision_tree_classifier_predict_proba():
    """
    predict_proba should return valid probability distributions
    """
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=2,
        random_state=0,
    )
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    assert proba.shape == (4, 2)
    # probabilities sum to 1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
    # non-negative
    assert np.all(proba >= 0.0)


def test_decision_tree_classifier_feature_importances():
    """
    feature_importances_ should exist and sum to 1 (or all zero)
    """
    rng = np.random.RandomState(0)
    X = rng.randn(50, 3)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    clf = DecisionTreeClassifier(
        max_depth=3,
        random_state=0,
    )
    clf.fit(X, y)

    importances = clf.feature_importances_

    assert importances is not None
    assert importances.shape == (3,)
    assert np.all(importances >= 0.0)

    total = importances.sum()
    assert np.isclose(total, 1.0) or np.isclose(total, 0.0)


def test_decision_tree_classifier_reproducibility():
    """
    random_state should make the model deterministic
    """
    rng = np.random.RandomState(1)
    X = rng.randn(100, 4)
    y = (X[:, 0] > 0).astype(int)

    clf1 = DecisionTreeClassifier(
        max_depth=4,
        max_features="sqrt",
        random_state=123,
    )
    clf2 = DecisionTreeClassifier(
        max_depth=4,
        max_features="sqrt",
        random_state=123,
    )

    clf1.fit(X, y)
    clf2.fit(X, y)

    pred1 = clf1.predict(X)
    pred2 = clf2.predict(X)

    assert np.array_equal(pred1, pred2)


# =========================================================
# Regression Tests
# =========================================================

def test_decision_tree_regressor_fit_predict_simple():
    """
    Simple 1D regression.
    CART predicts the mean value in each leaf (piecewise constant).
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    reg = DecisionTreeRegressor(
        max_depth=2,
        random_state=0,
    )
    reg.fit(X, y)
    y_pred = reg.predict(X)

    # shape check
    assert y_pred.shape == y.shape

    # CART behavior: piecewise constant prediction (leaf mean)
    expected = np.array([0.5, 0.5, 2.5, 2.5])
    np.testing.assert_allclose(y_pred, expected, rtol=1e-6)

    # error should be reasonably small
    mse = np.mean((y_pred - y) ** 2)
    assert mse < 0.3




def test_decision_tree_regressor_feature_importances():
    """
    feature_importances_ should exist for regression
    """
    rng = np.random.RandomState(0)
    X = rng.randn(80, 2)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.randn(80) * 0.01

    reg = DecisionTreeRegressor(
        max_depth=4,
        random_state=0,
    )
    reg.fit(X, y)

    importances = reg.feature_importances_

    assert importances is not None
    assert importances.shape == (2,)
    assert np.all(importances >= 0.0)

    total = importances.sum()
    assert np.isclose(total, 1.0) or np.isclose(total, 0.0)


def test_decision_tree_regressor_reproducibility():
    """
    random_state should make regression deterministic
    """
    rng = np.random.RandomState(42)
    X = rng.randn(120, 3)
    y = X[:, 0] - X[:, 1] + rng.randn(120) * 0.1

    reg1 = DecisionTreeRegressor(
        max_depth=5,
        max_features="log2",
        random_state=99,
    )
    reg2 = DecisionTreeRegressor(
        max_depth=5,
        max_features="log2",
        random_state=99,
    )

    reg1.fit(X, y)
    reg2.fit(X, y)

    pred1 = reg1.predict(X)
    pred2 = reg2.predict(X)

    np.testing.assert_allclose(pred1, pred2, rtol=1e-8)


# =========================================================
# Error Handling
# =========================================================

def test_decision_tree_classifier_invalid_input():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[0], [1]])  # wrong shape

    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_decision_tree_regressor_predict_before_fit():
    reg = DecisionTreeRegressor()
    X = np.array([[0.0]])

    with pytest.raises(ValueError):
        reg.predict(X)
