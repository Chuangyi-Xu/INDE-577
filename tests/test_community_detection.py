import numpy as np
import pytest

from rice_ml.community_detection import GirvanNewmanCommunityDetection


# ========== Helper Graphs ==========

def simple_two_community_graph():
    """
    Two clear communities:
    (0,1,2) -- fully connected
    (3,4,5) -- fully connected
    One weak bridge edge between 2 and 3
    """
    return {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3],
        3: [2, 4, 5],
        4: [3, 5],
        5: [3, 4],
    }


def triangle_graph():
    """
    Single community (triangle)
    """
    return {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
    }


# ========== Tests ==========

def test_fit_sets_communities_and_labels():
    """
    fit() should compute communities_, labels_, and modularity_
    """
    G = simple_two_community_graph()

    model = GirvanNewmanCommunityDetection(n_communities=2)
    model.fit(G)

    assert model.communities_ is not None
    assert model.labels_ is not None
    assert model.modularity_ is not None

    # Expect exactly 2 communities
    assert len(model.communities_) == 2

    # Every node should have a label
    for node in G.keys():
        assert node in model.labels_


def test_fit_predict_consistency():
    """
    fit_predict should be equivalent to fit + predict
    """
    G = simple_two_community_graph()

    model = GirvanNewmanCommunityDetection(n_communities=2)

    y1 = model.fit_predict(G)
    y2 = model.predict(G)

    assert isinstance(y1, np.ndarray)
    assert isinstance(y2, np.ndarray)
    assert y1.shape == y2.shape
    assert np.all(y1 == y2)


def test_predict_without_fit_raises_error():
    """
    predict before fit should raise an error
    """
    G = triangle_graph()

    model = GirvanNewmanCommunityDetection()

    with pytest.raises(RuntimeError):
        model.predict(G)


def test_single_community_graph():
    """
    Fully connected graph should result in one community
    """
    G = triangle_graph()

    model = GirvanNewmanCommunityDetection()
    model.fit(G)

    assert len(model.communities_) == 1

    labels = model.predict(G)
    assert np.all(labels == labels[0])


def test_edge_list_input():
    """
    Model should accept edge list input
    """
    edges = [
        (0, 1), (1, 2), (0, 2),   # community 1
        (3, 4), (4, 5), (3, 5),   # community 2
        (2, 3),                   # weak bridge
    ]

    model = GirvanNewmanCommunityDetection(n_communities=2)
    model.fit(edges)

    assert len(model.communities_) == 2
    labels = model.predict(edges)
    assert isinstance(labels, np.ndarray)


def test_adjacency_matrix_input():
    """
    Model should accept adjacency matrix input
    """
    A = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ])

    model = GirvanNewmanCommunityDetection(n_communities=2)
    model.fit(A)

    labels = model.predict(A)

    assert labels.shape[0] == A.shape[0]
    assert len(set(labels)) == 2


def test_unseen_node_in_predict():
    """
    Nodes not seen during fit should be labeled as -1
    """
    G_train = {
        0: [1],
        1: [0],
    }

    G_test = {
        0: [1],
        1: [0],
        2: [],   # unseen node
    }

    model = GirvanNewmanCommunityDetection()
    model.fit(G_train)

    labels = model.predict(G_test)

    assert labels[-1] == -1