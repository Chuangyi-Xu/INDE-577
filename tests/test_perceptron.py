# tests/test_perceptron.py
import numpy as np
from rice_ml.perceptron import Perceptron

def test_perceptron_fit_and_predict_simple_dataset():
    # AND logic gate dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0, 0, 0, 1])  # only (1,1) -> 1

    clf = Perceptron(max_iter=1000, lr=1.0, random_state=0)
    clf.fit(X, y)

    y_pred = clf.predict(X)

    # at least check shape and correctness
    assert y_pred.shape == y.shape
    assert np.array_equal(y_pred, y)

def test_perceptron_score_range():
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    clf = Perceptron(max_iter=10, lr=1.0, random_state=0)
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert 0.0 <= acc <= 1.0
