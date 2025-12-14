import numpy as np
from rice_ml.distance_metrics import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
)


def test_euclidean_distance_simple():
    x1 = np.array([0.0, 0.0])
    x2 = np.array([3.0, 4.0])
    assert euclidean_distance(x1, x2) == 5.0


def test_manhattan_distance_simple():
    x1 = np.array([1.0, 2.0])
    x2 = np.array([4.0, 6.0])
    assert manhattan_distance(x1, x2) == 7.0


def test_cosine_distance_orthogonal():
    x1 = np.array([1.0, 0.0])
    x2 = np.array([0.0, 1.0])
    assert np.isclose(cosine_distance(x1, x2), 1.0)


def test_distance_symmetry():
    x1 = np.array([2.0, 3.0])
    x2 = np.array([5.0, 7.0])
    assert euclidean_distance(x1, x2) == euclidean_distance(x2, x1)


def test_zero_distance():
    x = np.array([1.0, 2.0, 3.0])
    assert euclidean_distance(x, x) == 0.0
    assert manhattan_distance(x, x) == 0.0
    assert cosine_distance(x, x) == 0.0


def test_cosine_distance_zero_vector():
    x1 = np.array([0.0, 0.0])
    x2 = np.array([1.0, 1.0])
    assert cosine_distance(x1, x2) == 1.0
