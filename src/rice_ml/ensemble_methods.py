import numpy as np


class VotingClassifier:
    """
    Hard voting ensemble classifier.

    This classifier fits multiple base models and combines their
    predictions using majority voting.
    """

    def __init__(self, models):
        """
        Parameters
        ----------
        models : list
            List of initialized classification models.
            Each model must implement fit(X, y) and predict(X).
        """
        self.models = models

    def fit(self, X, y):
        """
        Fit all base models.
        """
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X using majority voting.
        """
        # Collect predictions from each model
        predictions = np.array([model.predict(X) for model in self.models])
        # shape: (n_models, n_samples)

        final_predictions = []

        for i in range(predictions.shape[1]):
            values, counts = np.unique(predictions[:, i], return_counts=True)
            final_predictions.append(values[np.argmax(counts)])

        return np.array(final_predictions)
