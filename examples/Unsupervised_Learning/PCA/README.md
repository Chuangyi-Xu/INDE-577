# DBSCAN Module

## 1. Introduction

This module implements the **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**  
algorithm from scratch using NumPy.

DBSCAN is an **unsupervised clustering** method that groups data points based on  
local point density, rather than assuming a predefined number of clusters or  
a specific cluster shape.

The implementation follows a clean, scikit-learn–style API and is integrated  
into the project with **unit tests** and a complete **Jupyter Notebook example**  
demonstrating clustering behavior and noise detection on real-world data.

---

## 2. Algorithm Overview

DBSCAN identifies clusters as contiguous regions of high point density and  
labels low-density points as noise.

Given two parameters:

- **ε (eps)**: neighborhood radius
    
- **min_samples**: minimum number of points required to form a dense region
    

the algorithm proceeds as follows:

1. For each data point, identify all neighboring points within distance ε.
    
2. Mark points with at least `min_samples` neighbors as **core points**.
    
3. Expand clusters by recursively connecting density-reachable core points.
    
4. Label points that are not density-reachable from any core point as **noise**.
    

Unlike centroid-based clustering methods, DBSCAN:

- Does not require specifying the number of clusters
    
- Can discover clusters of arbitrary shape
    
- Explicitly identifies noise and outliers
    

---

## 3. Dataset Description

The example notebook uses a real-world dataset obtained from Kaggle, consisting  
of multiple numerical features describing observations in a high-dimensional  
space.

Since DBSCAN is an **unsupervised algorithm**, no label information is required  
during clustering. All features are standardized prior to clustering to ensure  
that distance-based neighborhood queries are meaningful.

---

## 4. Implementation Details

- DBSCAN is implemented **from scratch using NumPy**, without relying on  
    `scikit-learn`.
    
- Pairwise distances are computed using configurable distance metrics.
    
- The algorithm explicitly tracks:
    
    - Core points
        
    - Border points
        
    - Noise points
        
- Cluster expansion is implemented using a breadth-first search (BFS)–style  
    procedure.
    
- The implementation follows a **scikit-learn–style API**, supporting:
    
    - `fit`
        
    - `fit_predict`
        

---

## 5. Experimental Results

Key observations from the DBSCAN clustering experiments include:

- The algorithm successfully identifies clusters without specifying the number  
    of clusters in advance.
    
- Noise points are automatically detected and labeled, improving robustness  
    to outliers.
    
- DBSCAN is able to recover clusters of non-spherical and irregular shapes,  
    which are difficult for centroid-based methods such as K-Means.
    

The results highlight the strengths of density-based clustering for exploratory  
data analysis.

---

## 6. Visualization

The accompanying Jupyter notebook includes:

- Visualization of clusters in two-dimensional space
    
- Clear distinction between core points, border points, and noise
    
- Sensitivity analysis with respect to `eps` and `min_samples`
    

These visualizations provide intuitive insight into how density-based clustering  
operates on real data.

---

## 7. Summary and Key Takeaways

- DBSCAN is a powerful density-based clustering algorithm that does not require  
    specifying the number of clusters.
    
- The algorithm naturally handles noise and outliers.
    
- Implementing DBSCAN from scratch highlights the role of neighborhood queries  
    and density connectivity.
    
- DBSCAN is well suited for exploratory data analysis and datasets with  
    irregular cluster structures.
    

---

## 8. Unit Tests

Unit tests ensure the correctness and robustness of the implementation:

- Model instantiation and parameter validation
    
- Correct identification of core, border, and noise points
    
- Consistent cluster labeling on simple synthetic datasets
    
- Stability of results under different parameter settings
    

Run tests:

`pytest tests/test_dbscan.py -q`