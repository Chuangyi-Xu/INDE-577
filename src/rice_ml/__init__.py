"""
rice_ml: A from-scratch machine learning library
Developed for INDE 577 / CMOR 438 at Rice University
"""

# Supervised Learning
from .perceptron import Perceptron
from .logistic_regression import LogisticRegression
from .linear_regression import LinearRegression
from .knn import KNN
from .decision_trees import DecisionTreeClassifier, DecisionTreeRegressor
from .random_forest import RandomForestClassifier, RandomForestRegressor
from .multilayer_perceptron import MultilayerPerceptron

# Unsupervised Learning
from .kmeans import KMeans
from .dbscan import DBSCAN
from .pca import PCA
from .label_propagation import LabelPropagation
from .community_detection import CommunityDetection

# Utilities
from .distance_metrics import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance
)

# Ensemble methods
from .ensemble_methods import (
    BaggingClassifier,
    VotingClassifier
)

__all__ = [
    # Supervised
    "Perceptron",
    "LogisticRegression",
    "LinearRegression",
    "KNN",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "MultilayerPerceptron",

    # Unsupervised
    "KMeans",
    "DBSCAN",
    "PCA",
    "LabelPropagation",
    "CommunityDetection",

    # Utilities
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",

    # Ensemble
    "BaggingClassifier",
    "VotingClassifier",
]
