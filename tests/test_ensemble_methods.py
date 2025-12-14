import numpy as np
from rice_ml.ensemble_methods import VotingClassifier
from rice_ml.knn import KNNClassifier


def test_voting_classifier_basic():
    """
    Test that VotingClassifier can fit and predict without errors.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    knn1 = KNNClassifier(k=1)
    knn2 = KNNClassifier(k=3)

    clf = VotingClassifier(models=[knn1, knn2])
    clf.fit(X, y)

    preds = clf.predict(X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)
