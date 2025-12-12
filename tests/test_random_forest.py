import numpy as np
import pytest

from rice_ml.random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
)


# =========================
# Classification Tests
# =========================

def test_random_forest_classifier_fit_predict_simple():
    """
    Simple binary classification.
    """
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0, 0, 1, 1])

    clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=2,
        random_state=0,
    )

    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert y_pred.shape == y.shape
    assert set(y_pred).issubset({0, 1})


def test_random_forest_classifier_perfect_separation():
    """
    Random Forest should perfectly fit separable data.
    """
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = RandomForestClassifier(
        n_estimators=20,
        max_depth=3,
        random_state=42,
    )

    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert np.array_equal(y_pred, y)


def test_random_forest_classifier_multiple_features():
    """
    Multi-dimensional input.
    """
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0, 1, 1, 0])  # XOR-like

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=1,
    )

    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert y_pred.shape == y.shape


# =========================
# Regression Tests
# =========================

def test_random_forest_regressor_fit_predict_simple():
    """
    Simple 1D regression.
    """
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    reg = RandomForestRegressor(
        n_estimators=20,
        max_depth=3,
        random_state=0,
    )

    reg.fit(X, y)
    y_pred = reg.predict(X)

    assert y_pred.shape == y.shape

    # performance check (Random Forest should approximate well)
    mse = np.mean((y - y_pred) ** 2)
    assert mse < 0.5



def test_random_forest_regressor_non_linear():
    """
    Non-linear regression.
    """
    X = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
    y = np.sin(X).ravel()

    reg = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        random_state=0,
    )

    reg.fit(X, y)
    y_pred = reg.predict(X)

    assert y_pred.shape == y.shape
    assert np.mean((y - y_pred) ** 2) < 0.1


# =========================
# Edge / API Tests
# =========================

def test_random_forest_predict_shape():
    """
    Output shape consistency.
    """
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, size=20)

    clf = RandomForestClassifier(
        n_estimators=5,
        random_state=0,
    )

    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert y_pred.ndim == 1
    assert len(y_pred) == X.shape[0]
