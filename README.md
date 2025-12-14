# INDE577 / CMOR 438 – Data Science & Machine Learning  
### Rice University · Fall 2025

By Chuangyi Xu

---

## Course Description

This repository contains homework implementations and learning materials for  
**INDE 577 / CMOR 438 – Data Science & Machine Learning** at Rice University.

The purpose of this repository is to demonstrate a full collection of machine
learning algorithms implemented **from scratch** using Python and NumPy.  
Each algorithm is accompanied by Jupyter notebook examples, visualizations,
and unit tests to support correctness and reproducibility.

This course covers supervised learning, unsupervised learning, optimization,
and practical machine learning workflows.

---

## Course Instructor

**Dr. Randy R. Davila**, Department of Computational & Applied Mathematics  
Rice University  

---

## Repository Description

This repository is organized into three major components:

1. **`src/rice_ml/`** –  
   Source code for all machine learning algorithms, written in a modular,
   scikit-learn–style API.

2. **`tests/`** –  
   Pytest unit tests ensuring correctness for each algorithm.

3. **`examples/`** –  
   Jupyter notebooks illustrating mathematical intuition, implementation
   details, and visual demonstrations for each model.

Programming language used: **Python 3**  
Package management: **pyproject.toml + pip editable install**

---
### Supervised Learning

- Perceptron
    
- Linear Regression
    
- Logistic Regression
    
- K-Nearest Neighbors (KNN)
    
- Decision Trees
    
- Regression Trees
    
- Random Forests
    
- Ensemble Methods (Bagging / Voting)
    
- Multilayer Perceptron (MLP)
    

### Unsupervised Learning

- Principal Component Analysis (PCA)
    
- K-Means Clustering
    
- DBSCAN
    
- Label Propagation
    
- Community Detection
    

### Optimization Concepts (Used in Models)

- Gradient Descent
    
- Mini-batch Gradient Descent
    
- Regularization Techniques
    

### Model Comparison

- Experimental evaluation through notebooks
    
- Visual analysis and metrics
    
- Hyperparameter effects

---

## Repository Structure

```
ml_project/
│  .gitignore
│  LICENSE
│  pyproject.toml
│  README.md
│  
├─.github
│  └─ISSUE_TEMPLATE
│          feature_request.md
│
├─.pytest_cache
│  │  .gitignore
│  │  CACHEDIR.TAG
│  │  README.md
│  │  
│  └─v
│      └─cache
│              lastfailed
│              nodeids
│
├─examples
│  ├─data
│  │      auto-mpg.csv
│  │      bank-additional-small.csv
│  │      Iris.csv
│  │      KNN_data.csv
│  │      Mall_Customers.csv
│  │      Mining_small.csv
│  │      pca_data.csv
│  │      regressiontree_submission.csv
│  │      SeoulBikeData.csv
│  │      time_series_data_human_activities.csv
│  │      top_insta_influencers_data.csv
│  │      Wholesale_customers_data.csv
│  │      WineQT.csv
│  │
│  ├─Supervised_Learning
│  │  ├─Decision_Trees
│  │  │      Decision_Trees.ipynb
│  │  │      README.md
│  │  │
│  │  ├─Ensemble_Methods
│  │  │      Ensemble_Methods.ipynb
│  │  │      README.md
│  │  │
│  │  ├─KNN
│  │  │      KNN.ipynb
│  │  │      README.md
│  │  │
│  │  ├─Linear_Regression
│  │  │      Linear_Regression.ipynb
│  │  │      README.md
│  │  │      
│  │  ├─Logistic_Regression
│  │  │      Logistic_Regression.ipynb
│  │  │      README.md
│  │  │
│  │  ├─Multilayer_Perceptron
│  │  │      Multilayer_Perceptron.ipynb
│  │  │      README.md
│  │  │
│  │  ├─Perceptron
│  │  │      Perceptron.ipynb
│  │  │      README.md
│  │  │
│  │  ├─Random_Forests
│  │  │      Random_Forests.ipynb
│  │  │      README.md
│  │  │
│  │  └─Regression_Trees
│  │          README.md
│  │          Regression_Trees.ipynb
│  │
│  └─Unsupervised_Learning
│      ├─Community_Detection
│      │      Community_Detection.ipynb
│      │      README.md
│      │
│      ├─DBSCAN
│      │      DBSCAN.ipynb
│      │      README.md
│      │
│      ├─KMeans
│      │      KMeans.ipynb
│      │      README.md
│      │
│      ├─Label_Propagation
│      │      Label_Propagation.ipynb
│      │      README.md
│      │
│      └─PCA
│              PCA.ipynb
│              README.md
│
├─src
│  ├─rice_ml
│  │  │  community_detection.py
│  │  │  dbscan.py
│  │  │  decision_trees.py
│  │  │  distance_metrics.py
│  │  │  ensemble_methods.py
│  │  │  kmeans.py
│  │  │  knn.py
│  │  │  label_propagation.py
│  │  │  linear_regression.py
│  │  │  logistic_regression.py
│  │  │  multilayer_perceptron.py
│  │  │  pca.py
│  │  │  perceptron.py
│  │  │  random_forest.py
│  │  │  regression_trees.py
│  │  │  __init__.py
│  │  │
│  │  └─__pycache__
│  │          community_detection.cpython-39.pyc
│  │          dbscan.cpython-39.pyc
│  │          decision_trees.cpython-39.pyc
│  │          distance_metrics.cpython-39.pyc
│  │          ensemble_methods.cpython-39.pyc
│  │          kmeans.cpython-39.pyc
│  │          knn.cpython-39.pyc
│  │          label_propagation.cpython-39.pyc
│  │          linear_regression.cpython-39.pyc
│  │          logistic_regression.cpython-39.pyc
│  │          multilayer_perceptron.cpython-39.pyc
│  │          pca.cpython-39.pyc
│  │          perceptron.cpython-39.pyc
│  │          random_forest.cpython-39.pyc
│  │          regression_trees.cpython-39.pyc
│  │          __init__.cpython-39.pyc
│  │
│  └─rice_ml.egg-info
│          dependency_links.txt
│          PKG-INFO
│          requires.txt
│          SOURCES.txt
│          top_level.txt
│
└─tests
    │  test_community_detection.py
    │  test_dbscan.py
    │  test_decision_trees.py
    │  test_distance_metrics.py
    │  test_ensemble_methods.py
    │  test_kmeans.py
    │  test_knn.py
    │  test_label_propagation.py
    │  test_linear_regression.py
    │  test_logistic_regression.py
    │  test_multilayer_perceptron.py
    │  test_pca.py
    │  test_perceptron.py
    │  test_random_forest.py
    │  test_regression_trees.py
    │
    └─__pycache__
            test_community_detection.cpython-39-pytest-8.4.2.pyc
            test_dbscan.cpython-39-pytest-8.4.2.pyc
            test_decision_trees.cpython-39-pytest-8.4.2.pyc
            test_distance_metrics.cpython-39-pytest-8.4.2.pyc
            test_ensemble_methods.cpython-39-pytest-8.4.2.pyc
            test_kmeans.cpython-39-pytest-8.4.2.pyc
            test_knn.cpython-39-pytest-8.4.2.pyc
            test_label_propagation.cpython-39-pytest-8.4.2.pyc
            test_linear_regression.cpython-39-pytest-8.4.2.pyc
            test_logistic_regression.cpython-39-pytest-8.4.2.pyc
            test_multilayer_perceptron.cpython-39-pytest-8.4.2.pyc
            test_pca.cpython-39-pytest-8.4.2.pyc
            test_perceptron.cpython-39-pytest-8.4.2.pyc
            test_random_forest.cpython-39-pytest-8.4.2.pyc
            test_regression_trees.cpython-39-pytest-8.4.2.pyc
```

---

## Installation

To install this project in editable mode, navigate to the project root and run:

```
pip install -e .
```

This allows you to import the package as:

```python
from rice_ml.knn import KNN
```

---

## Running Tests

Unit tests are located in the `tests/` directory.

Run all tests:

```
pytest -q
```

Run a specific test module (example):

```
pytest tests/test_perceptron.py -q
```

---

## Running Examples (Jupyter Notebooks)

All example notebooks are located in the `examples/` directory.

To run the notebooks locally:

Clone the repository:
```
   git clone https://github.com/Chuangyi-Xu/INDE-577.git
   cd INDE-577
```
---

## License

This repository is released under the MIT License.  
See the `LICENSE` file for details.

---

## Acknowledgements

This repository was created as part of  
**INDE 577 / CMOR 438 – Data Science & Machine Learning**  
at **Rice University**, instructed by **Dr. Randy R. Davila**.
