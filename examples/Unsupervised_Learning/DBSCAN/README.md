# DBSCAN Module

This module implements a **density-based clustering algorithm (DBSCAN)** from scratch using NumPy.  
It follows a clean, scikit-learn–style API and is fully integrated into the project with unit tests and a complete Jupyter Notebook example based on a real-world customer segmentation dataset.

The implementation focuses on clarity, correctness, and educational value, while remaining suitable for reuse as part of the `rice_ml` package.

---

## Features

- Density-based clustering without requiring the number of clusters in advance
    
- Explicit identification of noise (outliers)
    
- Support for arbitrary-shaped clusters
    
- Support for key hyperparameters:
    
    - `eps`
        
    - `min_samples`
        
- Fully deterministic behavior given fixed inputs
    
- Exposes learned attributes:
    
    - `labels_`
        
    - `core_sample_indices_`
        
    - `components_`
        
- Provides:
    
    - `fit()` for model training
        
    - `fit_predict()` for direct clustering
        
- Fully compatible with the project testing framework (`pytest`)
    

---

## File Structure

- `src/rice_ml/dbscan.py`  
    Core DBSCAN implementation
    
- `tests/test_dbscan.py`  
    Unit tests validating correctness and edge cases
    
- `examples/Unsupervised_Learning/DBSCAN/DBSCAN.ipynb`  
    End-to-end notebook example using a real dataset
    

---

## Class API

The DBSCAN implementation follows a scikit-learn–style interface and can be used interchangeably with other clustering modules in the project.

Key design goals of the API include:

- Clear separation between model initialization and fitting
    
- Consistent attribute naming across clustering algorithms
    
- Compatibility with standardized preprocessing workflows
    

---

## Notebook Overview — Real-World Application

The example notebook demonstrates DBSCAN applied to a **customer segmentation problem** using the Mall Customers dataset from Kaggle.

The notebook provides a complete unsupervised learning workflow, including:

1. Data loading and inspection
    
2. Feature selection for clustering
    
3. Feature scaling and preprocessing
    
4. Application of DBSCAN clustering
    
5. Visualization of clusters and noise points
    
6. Interpretation of clustering results
    

This example highlights DBSCAN’s ability to identify dense customer groups while explicitly isolating outliers.

---

## Key Results

- DBSCAN successfully discovers multiple customer groups without specifying the number of clusters
    
- Dense regions of customers are identified as clusters of varying sizes
    
- Customers that do not belong to any dense region are classified as noise
    
- The resulting clusters reflect realistic and irregular customer behavior patterns
    
- Feature scaling is shown to be critical for meaningful density-based clustering
    

---

## Unit Tests

Unit tests ensure the correctness and robustness of the implementation, including:

- Model instantiation with valid and invalid parameters
    
- Correct identification of clusters and noise points
    
- Validation of core sample detection
    
- Handling of edge cases such as empty inputs and all-noise datasets
    
- Consistency of `fit()` and `fit_predict()` behavior
    

Tests can be executed using the project’s standard testing workflow.

---

## Reference Notebook

The complete example is available at:

- `examples/Unsupervised_Learning/DBSCAN/DBSCAN.ipynb`
    

The notebook provides a full workflow from raw customer data to clustering visualization and interpretation.

---

## Summary

This DBSCAN module serves as:

- A clean educational implementation of density-based clustering
    
- A reusable unsupervised learning component within the `rice_ml` package
    
- A practical example of applying clustering to real-world customer data
    
- A foundation for future density-based and graph-based clustering methods