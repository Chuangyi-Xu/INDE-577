import numpy as np


def euclidean_distance(x1, x2):
    """
    Compute the Euclidean (L2) distance between two vectors.

    Parameters
    ----------
    x1 : np.ndarray
        First input vector.
    x2 : np.ndarray
        Second input vector.

    Returns
    -------
    float
        Euclidean distance between x1 and x2.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    """
    Compute the Manhattan (L1) distance between two vectors.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.sum(np.abs(x1 - x2))


def cosine_distance(x1, x2):
    """
    Compute the cosine distance between two vectors.

    Cosine distance = 1 - cosine similarity.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    numerator = np.dot(x1, x2)
    denominator = np.linalg.norm(x1) * np.linalg.norm(x2)

    if denominator == 0:
        return 1.0

    return 1.0 - numerator / denominator
