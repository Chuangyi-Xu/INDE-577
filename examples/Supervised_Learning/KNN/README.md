# K-Nearest Neighbors (KNN) Module

This module implements a **k-Nearest Neighbors (KNN) classifier from scratch**  
using NumPy. The implementation follows a clean, scikit-learn–style API and is  
fully integrated into the project with **unit tests** and a complete **Jupyter  
Notebook example** based on a real-world Kaggle dataset.

KNN is a non-parametric, instance-based learning algorithm that makes predictions  
based on similarity in feature space, providing a strong contrast to  
optimization-based models such as logistic regression.

---

## Features

- Distance-based, non-parametric classification
    
- Support for multiple distance metrics:
    
    - `euclidean`
        
    - `manhattan`
        
- Configurable number of neighbors (`k`)
    
- Clean `fit()` / `predict()` API
    
- Fully vectorized distance computation
    
- Compatible with the project testing framework (`pytest`)
    

---

## Class API
```
from rice_ml.knn import KNNClassifier  

clf = KNNClassifier(     
	k=5,     
	metric="euclidean" 
)  

clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)`

```
### Parameters

- `k` : int  
    Number of nearest neighbors used for majority voting.
    
- `metric` : str  
    Distance metric used to compute similarity between samples.  
    Supported values: `"euclidean"`, `"manhattan"`.
    

---

## Notebook Overview — Real-World Classification

The example notebook demonstrates the application of the KNN algorithm to the  
**Breast Cancer Wisconsin (Diagnostic)** dataset obtained from Kaggle. The goal  
is to classify tumors as **malignant** or **benign** based on numerical features  
extracted from medical imaging data.

### Notebook includes:

1. Data loading and inspection
    
2. Target variable encoding (malignant vs. benign)
    
3. Feature selection and cleanup
    
4. Feature standardization (crucial for KNN)
    
5. Train–test split
    
6. Training a baseline KNN classifier
    
7. Model evaluation:
    
    - Classification accuracy
        
8. Hyperparameter analysis:
    
    - Effect of the number of neighbors $k$
        
    - Bias–variance trade-off visualization
        

---

## Key Results

- KNN achieves **high classification accuracy** on the test set after feature  
    standardization.
    
- Model performance is sensitive to the choice of the number of neighbors.
    
- An intermediate value of $k$ provides the best balance between bias and  
    variance.
    
- The results empirically demonstrate the theoretical behavior of KNN as a  
    distance-based learning algorithm.
    

---

## Unit Tests

Unit tests ensure the correctness and robustness of the implementation:

- Model initialization with valid and invalid parameters
    
- Correct behavior of `fit()` and `predict()`
    
- Validation of distance metric handling
    
- Correct predictions on simple synthetic datasets
    

Run tests with:

`pytest tests/test_knn.py -q`

---

## Summary

This module provides a clear and complete implementation of the k-Nearest  
Neighbors algorithm, highlighting the strengths and limitations of  
distance-based learning. Together with the accompanying notebook and tests,  
it serves as a practical and educational example of non-parametric  
classification methods in machine learning.