import numpy as np
from rice_ml.linear_regression import LinearRegression


def test_linear_regression_fit():
    # Simple linear relationship: y = 3x + 2
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([5, 8, 11, 14, 17])

    model = LinearRegression(lr=0.01, num_iter=5000, tol=1e-7)
    model.fit(X, y)

    # check weights close to 3
    assert np.isclose(model.weights_[0], 3, atol=0.1)

    # check bias close to 2
    assert np.isclose(model.bias_, 2, atol=0.1)


def test_linear_regression_predict():
    X_train = np.array([[0], [1], [2]])
    y_train = np.array([1, 3, 5])   # y = 2x + 1

    model = LinearRegression(lr=0.01, num_iter=3000)
    model.fit(X_train, y_train)

    # predict on new values
    X_test = np.array([[3], [4]])
    preds = model.predict(X_test)

    # expected y = 2x + 1 => [7, 9]
    assert np.allclose(preds, np.array([7, 9]), atol=0.2)


def test_linear_regression_score():
    # Perfect linear data: y = -4x + 10
    X = np.array([[1], [2], [3], [4], [5]])
    y = 10 - 4 * X.reshape(-1)

    model = LinearRegression(lr=0.01, num_iter=6000)
    model.fit(X, y)

    r2 = model.score(X, y)

    # should be very close to 1
    assert r2 > 0.98