import numpy as np
import pytest

from rice_ml.regression_trees import RegressionTreeRegressor


# =========================
# Basic functionality
# =========================

def test_regression_tree_fit_predict_simple():
    """
    Perfectly separable 1D regression.
    Tree should fit training data exactly with enough depth.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = RegressionTreeRegressor(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=0,
    )

    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == y.shape
    assert np.allclose(y_pred, y, atol=1e-8)


def test_regression_tree_predict_shape():
    """
    Predict output shape should be (n_samples,).
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)

    model = RegressionTreeRegressor(random_state=0)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == (20,)


# =========================
# Stopping criteria
# =========================

def test_max_depth_limits_tree():
    """
    With max_depth=0, the tree should be a single leaf
    predicting the mean of y.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    model = RegressionTreeRegressor(max_depth=0)
    model.fit(X, y)

    y_pred = model.predict(X)
    expected = np.mean(y)

    assert np.allclose(y_pred, expected)


def test_min_samples_leaf_enforced():
    """
    min_samples_leaf should prevent overly small leaves.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 10.0, 10.0])

    model = RegressionTreeRegressor(
        min_samples_leaf=3,
        random_state=0,
    )
    model.fit(X, y)

    # Cannot split into two leaves of size >= 3
    # so prediction should be global mean
    y_pred = model.predict(X)
    assert np.allclose(y_pred, np.mean(y))


# =========================
# Feature handling
# =========================

def test_max_features_int():
    """
    max_features as int should limit number of features considered.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 5))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=50)

    model = RegressionTreeRegressor(
        max_features=1,
        random_state=0,
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == (50,)


def test_max_features_sqrt():
    """
    max_features="sqrt" should run without error.
    """
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 9))
    y = rng.normal(size=30)

    model = RegressionTreeRegressor(
        max_features="sqrt",
        random_state=0,
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == (30,)


# =========================
# Feature importances
# =========================

def test_feature_importances_sum_to_one():
    """
    Feature importances should sum to 1 after fitting
    (unless tree never splits).
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = 3 * X[:, 2] + rng.normal(scale=0.1, size=100)

    model = RegressionTreeRegressor(random_state=0)
    model.fit(X, y)

    importances = model.feature_importances_
    assert importances.shape == (4,)
    assert np.isclose(importances.sum(), 1.0)


def test_feature_importance_identifies_signal():
    """
    Feature with strongest signal should have highest importance.
    """
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3))
    y = 5 * X[:, 1] + rng.normal(scale=0.05, size=200)

    model = RegressionTreeRegressor(
        max_depth=3,
        random_state=0,
    )
    model.fit(X, y)

    importances = model.feature_importances_
    assert np.argmax(importances) == 1


# =========================
# Input validation
# =========================

def test_predict_before_fit_raises():
    """
    Calling predict before fit should raise an error.
    """
    model = RegressionTreeRegressor()
    X = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError):
        model.predict(X)


def test_invalid_input_shapes():
    """
    Invalid X / y shapes should raise ValueError.
    """
    model = RegressionTreeRegressor()

    X = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_nan_input_raises():
    """
    NaNs in X or y should raise ValueError.
    """
    X = np.array([[0.0], [1.0], [np.nan]])
    y = np.array([0.0, 1.0, 2.0])

    model = RegressionTreeRegressor()

    with pytest.raises(ValueError):
        model.fit(X, y)
