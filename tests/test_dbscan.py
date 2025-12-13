import numpy as np
import pytest

from rice_ml.dbscan import DBSCAN


def test_dbscan_two_clear_clusters():
    """
    Two well-separated clusters + no noise
    """
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [5.0, 5.0],
        [5.1, 5.0],
        [5.0, 5.1],
    ])

    db = DBSCAN(eps=0.3, min_samples=2)
    labels = db.fit_predict(X)

    # should find exactly 2 clusters
    unique_labels = set(labels)
    assert -1 not in unique_labels
    assert len(unique_labels) == 2

    # points 0,1,2 same cluster
    assert labels[0] == labels[1] == labels[2]
    # points 3,4,5 same cluster
    assert labels[3] == labels[4] == labels[5]
    # clusters are different
    assert labels[0] != labels[3]


def test_dbscan_with_noise_points():
    """
    Cluster with noise
    """
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],  # noise
        [20.0, 20.0],  # noise
    ])

    db = DBSCAN(eps=0.3, min_samples=3)
    labels = db.fit_predict(X)

    # first three points form a cluster
    assert labels[0] == labels[1] == labels[2]
    # last two are noise
    assert labels[3] == -1
    assert labels[4] == -1


def test_dbscan_all_noise():
    """
    All points are noise
    """
    X = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
        [20.0, 20.0],
    ])

    db = DBSCAN(eps=0.5, min_samples=2)
    labels = db.fit_predict(X)

    assert np.all(labels == -1)


def test_dbscan_min_samples_one():
    """
    min_samples = 1 => every point is a core point
    """
    X = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
        [20.0, 20.0],
    ])

    db = DBSCAN(eps=0.1, min_samples=1)
    labels = db.fit_predict(X)

    # each point becomes its own cluster
    assert len(set(labels)) == 3
    assert -1 not in labels


def test_dbscan_core_sample_indices():
    """
    core_sample_indices_ correctness
    """
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [5.0, 5.0],
    ])

    db = DBSCAN(eps=0.3, min_samples=3)
    db.fit(X)

    core_idx = db.core_sample_indices_

    # first three points are core
    assert set(core_idx.tolist()) == {0, 1, 2}
    # components_ matches core points
    assert db.components_.shape[0] == 3


def test_dbscan_invalid_eps():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError):
        DBSCAN(eps=0.0).fit(X)

    with pytest.raises(ValueError):
        DBSCAN(eps=-1.0).fit(X)


def test_dbscan_invalid_min_samples():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError):
        DBSCAN(min_samples=0).fit(X)


def test_dbscan_empty_input():
    """
    Empty dataset should not crash
    """
    X = np.empty((0, 2))

    db = DBSCAN()
    labels = db.fit_predict(X)

    assert labels.size == 0
    assert db.core_sample_indices_.size == 0
    assert db.components_.shape[0] == 0
