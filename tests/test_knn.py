import numpy as np
from rice_ml.knn import KNNClassifier


def test_knn_fit_predict_binary():
    # Simple dataset (linearly separable)
    X_train = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [4.0, 4.0],
        [4.1, 3.9],
    ])
    y_train = np.array([0, 0, 1, 1])

    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    X_test = np.array([
        [1.1, 1.2],   # closer to class 0
        [4.2, 3.8],   # closer to class 1
    ])

    y_pred = model.predict(X_test)

    assert y_pred[0] == 0
    assert y_pred[1] == 1


def test_knn_manhattan_distance():
    X_train = np.array([
        [0, 0],
        [3, 3]
    ])
    y_train = np.array([0, 1])

    model = KNNClassifier(k=1, metric="manhattan")
    model.fit(X_train, y_train)

    X_test = np.array([
        [1, 1],  # Manhattan distances: to [0,0] = 2, to [3,3] = 4 → class 0
        [2, 2],  # distances: 4 vs 2 → class 1
    ])

    y_pred = model.predict(X_test)

    assert y_pred[0] == 0
    assert y_pred[1] == 1


def test_knn_k_value():
    """Check that model raises error when k < 1."""
    try:
        KNNClassifier(k=0)
        assert False  # Should not reach here
    except ValueError:
        assert True
