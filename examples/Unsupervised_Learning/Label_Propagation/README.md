# Label Propagation Module

This module implements a **graph-based label propagation algorithm** for semi-supervised learning from scratch using NumPy.  
It follows a clean, **scikit-learn–style API** and is fully integrated into the project with **unit tests** and a complete **Jupyter Notebook example** based on a human activity recognition dataset.

Label propagation leverages the intrinsic structure of the data by constructing a similarity graph and propagating label information from a small set of labeled samples to a larger set of unlabeled samples.

---

## Features

- Graph-based semi-supervised learning
    
- RBF (Gaussian) similarity kernel
    
- Iterative label propagation with convergence control
    
- Support for:
    
    - `gamma` — RBF kernel coefficient
        
    - `max_iter` — maximum number of propagation iterations
        
    - `tol` — convergence tolerance
        
- Preserves labeled data through label clamping
    
- Returns soft label distributions via `label_distributions_`
    
- Includes:
    
    - `fit()` — build similarity graph and propagate labels
        
    - `predict()` — infer labels for all samples
        
    - `fit_predict()` — combined training and prediction
        
- Fully compatible with the project testing framework (`pytest`)
    

---

## Class API
```
from rice_ml.label_propagation import LabelPropagation  

lp = LabelPropagation(     
	gamma=1.0,     
	max_iter=100,     
	tol=1e-3 
)  
lp.fit(X, y_semi) 
y_pred = lp.predict()`
```
or equivalently:

`y_pred = lp.fit_predict(X, y_semi)`

Unlabeled samples should be marked with the value `-1`.

---

## Notebook Overview — Human Activity Recognition

The example notebook demonstrates **label propagation for human activity recognition** using wearable sensor data obtained from Kaggle.

Each sample consists of tri-axial sensor measurements (`x-axis`, `y-axis`, `z-axis`), along with a corresponding activity label (e.g., walking, sitting, standing).  
In realistic settings, labeling such data is expensive and time-consuming, making semi-supervised learning a natural choice.

### Notebook includes:

1. Data loading from CSV
    
2. Feature selection and removal of non-feature columns
    
3. Feature standardization for distance-based similarity
    
4. Subsampling for computational feasibility
    
5. Construction of a semi-supervised learning setting
    
6. Training a label propagation model
    
7. Label inference for unlabeled samples
    
8. 2D visualization using PCA
    

---

## Key Results

- Label propagation successfully infers activity labels using only a small fraction of labeled samples
    
- The algorithm exploits the geometric structure of the sensor data
    
- PCA visualizations show coherent activity clusters after propagation
    
- Demonstrates the effectiveness of graph-based semi-supervised learning on real-world time-series data
    

---

## Unit Tests

Unit tests ensure the correctness and robustness of the implementation:

- Model instantiation
    
- Proper handling of labeled and unlabeled samples
    
- Correct output shapes for predictions
    
- Preservation of labeled data during propagation
    
- Basic propagation behavior on synthetic datasets
    

Run tests with:

`pytest tests/test_label_propagation.py -q`

---

## Summary

This module provides a clean and interpretable implementation of label propagation, showcasing how semi-supervised learning can effectively leverage unlabeled data.  
It complements the supervised and unsupervised algorithms in the project and highlights the power of graph-based learning methods in practical applications.