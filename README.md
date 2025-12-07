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

## Implemented Topics

### **Supervised Learning**
- Perceptron  
- Linear Regression  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Trees  
- Random Forests  
- Ensemble Methods (Bagging / Boosting)  
- Multilayer Perceptron (MLP)

### **Unsupervised Learning**
- Principal Component Analysis (PCA)  
- K-Means Clustering  
- DBSCAN  
- Label Propagation  

### **Optimization Concepts (Used in Models)**
- Gradient Descent  
- Mini-batch Gradient Descent  
- Regularization Techniques  

### **Model Comparison**
- Experimental evaluation through notebooks  
- Visual analysis and metrics  
- Hyperparameter effects  

---

## Repository Structure

```
ml_project/
│
├── LICENSE
├── README.md
├── pyproject.toml
├── .gitignore
│
├── src/
│   └── rice_ml/
│       ├── __init__.py
│       ├── perceptron.py
│       ├── logistic_regression.py
│       ├── linear_regression.py
│       ├── multilayer_perceptron.py
│       ├── knn.py
│       ├── decision_trees.py
│       ├── random_forest.py
│       ├── ensemble_methods.py
│       ├── dimensionality_reduction.py
│       ├── kmeans.py
│       ├── dbscan.py
│       └── label_propagation.py
│
├── tests/
│       ├── test_perceptron.py
│       ├── test_logistic_regression.py
│       ├── test_linear_regression.py
│       ├── test_multilayer_perceptron.py
│       ├── test_knn.py
│       ├── test_decision_trees.py
│       ├── test_random_forest.py
│       ├── test_ensemble_methods.py
│       ├── test_dimensionality_reduction.py
│       ├── test_kmeans.py
│       ├── test_dbscan.py
│       └── test_label_propagation.py
│
└── examples/
        ├── Supervised_Learning/
        │       ├── Perceptron.ipynb
        │       ├── Logistic_Regression.ipynb
        │       ├── Linear_Regression.ipynb
        │       ├── Multilayer_Perceptron.ipynb
        │       ├── KNN.ipynb
        │       ├── Decision_Trees.ipynb
        │       ├── Random_Forests.ipynb
        │       └── Ensemble_Methods.ipynb
        │
        └── Unsupervised_Learning/
                ├── PCA.ipynb
                ├── KMeans.ipynb
                ├── DBSCAN.ipynb
                └── Label_Propagation.ipynb
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

## License

This repository is released under the MIT License.  
See the `LICENSE` file for details.

---

## Acknowledgements

This repository was created as part of  
**INDE 577 / CMOR 438 – Data Science & Machine Learning**  
at **Rice University**, instructed by **Dr. Randy R. Davila**.
