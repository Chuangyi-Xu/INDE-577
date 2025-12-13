# K-Means Clustering Module

This module implements the **K-Means clustering algorithm from scratch** using NumPy.  
It follows a clean, scikit-learn–style API and is fully integrated into the project with  
**unit tests** and a complete **Jupyter Notebook example** based on a real-world customer dataset.

The example demonstrates how K-Means can be applied to unsupervised learning tasks,  
including preprocessing, clustering, visualization, and model selection.

---

## Features

- Fully from-scratch K-Means implementation (no `sklearn.cluster.KMeans`)
    
- Iterative assignment and centroid update steps
    
- Support for:
    
    - `n_clusters`
        
    - `max_iter`
        
    - `tol`
        
    - `random_state`
        
- Tracks clustering quality through:
    
    - `inertia_` (within-cluster sum of squared distances)
        
- Returns learned model attributes:
    
    - `cluster_centers_`
        
    - `labels_`
        
- Includes:
    
    - `fit()` — train the clustering model
        
    - `predict()` — assign clusters to new data
        
    - `fit_predict()` — train and return labels
        
- Fully compatible with the project testing framework (`pytest`)
    

---
## Class API

```
from rice_ml.kmeans import KMeans  
kmeans = KMeans(     
	n_clusters=3,     
	max_iter=300,     
	tol=1e-4,     
	random_state=42, 
)  

labels = kmeans.fit_predict(X)`
```

After fitting, the following attributes are available:

- `kmeans.cluster_centers_`
    
- `kmeans.labels_`
    
- `kmeans.inertia_`
    

---

## Notebook Overview — Unsupervised Learning Example

The example notebook demonstrates K-Means clustering on a **real-world wholesale  
customer spending dataset**.

The goal is to group customers with similar purchasing behavior using unsupervised  
learning, without relying on labeled data.

### Notebook includes:

1. Data loading and basic inspection
    
2. Feature selection for clustering
    
3. Feature scaling for distance-based learning
    
4. Training K-Means using a custom implementation
    
5. Visualization of clustering results using PCA (2D projection)
    
6. Model selection using the Elbow Method
    
7. Summary and key observations
    

PCA is used **only for visualization purposes** and does not affect the clustering  
results.

---

## Key Results

- K-Means successfully groups customers based on spending behavior
    
- Feature scaling significantly improves clustering stability
    
- PCA visualization reveals clear cluster separation in reduced dimensions
    
- The Elbow Method provides a principled way to choose the number of clusters KKK
    

This example highlights the importance of preprocessing and model selection in  
distance-based unsupervised learning algorithms.

---

## Unit Tests

Unit tests ensure the correctness and robustness of the implementation, including:

- Model instantiation and attribute initialization
    
- Correct behavior of `fit`, `predict`, and `fit_predict`
    
- Reproducibility with a fixed `random_state`
    
- Proper error handling for invalid inputs
    
- Verification of output shapes and data types
    

Run tests with:

`pytest tests/test_kmeans.py -q`

---

## Summary

This module provides a clean and educational implementation of K-Means clustering,  
demonstrating both the algorithmic details and practical considerations required for  
real-world unsupervised learning tasks.

It is designed to be consistent with other modules in this project and serves as a  
foundation for more advanced clustering and ensemble techniques.