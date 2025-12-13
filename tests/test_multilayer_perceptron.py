import numpy as np
import pytest

from rice_ml.multilayer_perceptron import MLPClassifier, MLPRegressor


def test_mlp_classifier_binary_fit_predict_xor():
    # XOR (non-linear separable)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    clf = MLPClassifier(
        hidden_layer_sizes=(8, 8),
        activation="tanh",
        learning_rate=0.1,
        max_iter=4000,
        batch_size=4,
        l2=0.0,
        random_state=0,
        tol=1e-8,
        n_iter_no_change=200,
        shuffle=True,
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    acc = np.mean(pred == y)
    assert acc >= 0.95


def test_mlp_classifier_predict_proba_sums_to_one():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = rng.randint(0, 3, size=50)

    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        learning_rate=0.05,
        max_iter=800,
        batch_size=16,
        random_state=0,
    )
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (50, 3)
    row_sums = np.sum(proba, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_mlp_classifier_reproducible_with_random_state():
    rng = np.random.RandomState(1)
    X = rng.randn(80, 5)
    y = (rng.rand(80) > 0.5).astype(int)

    clf1 = MLPClassifier(hidden_layer_sizes=(10,), random_state=42, max_iter=300, learning_rate=0.05)
    clf2 = MLPClassifier(hidden_layer_sizes=(10,), random_state=42, max_iter=300, learning_rate=0.05)

    clf1.fit(X, y)
    clf2.fit(X, y)

    p1 = clf1.predict(X)
    p2 = clf2.predict(X)
    assert np.array_equal(p1, p2)


def test_mlp_regressor_fit_simple_line():
    # y = 2x + 1
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2 * X.ravel() + 1.0

    reg = MLPRegressor(
        hidden_layer_sizes=(16, 16),
        activation="tanh",
        learning_rate=0.05,
        max_iter=2000,
        batch_size=32,
        l2=0.0,
        random_state=0,
        tol=1e-8,
        n_iter_no_change=200,
    )
    reg.fit(X, y)

    y_pred = reg.predict(X)
    mse = np.mean((y_pred - y) ** 2)
    assert mse < 1e-2


def test_mlp_loss_curve_exists_and_decreases_somewhat():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 3)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    clf = MLPClassifier(
        hidden_layer_sizes=(12,),
        activation="relu",
        learning_rate=0.05,
        max_iter=400,
        batch_size=32,
        random_state=0,
        tol=1e-10,
        n_iter_no_change=1000,  # avoid early stop
    )
    clf.fit(X, y)
    assert len(clf.loss_curve_) > 5
    assert clf.loss_curve_[0] >= clf.loss_curve_[-1]
