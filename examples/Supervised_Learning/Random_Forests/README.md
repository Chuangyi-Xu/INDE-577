# Random Forest Module

This module implements a **Random Forest learning algorithm from scratch** using NumPy, following a clean, **scikit-learn–style API**.  
The implementation builds upon a custom Decision Tree base learner and demonstrates how **ensemble learning (bagging)** improves model stability and generalization.

The module is fully integrated into the project with **unit tests** and a complete **Jupyter Notebook example** using a real-world bike sharing demand dataset.

---

## Features

- Ensemble learning via **bootstrap aggregation (bagging)**
    
- Support for:
    
    - `n_estimators`
        
    - `max_depth`
        
    - `min_samples_split`
        
    - `max_features`
        
    - `bootstrap`
        
    - `random_state`
        
- Supports both:
    
    - **Regression** (prediction averaging)
        
    - **Classification** (majority voting)
        
- Reuses custom `DecisionTreeRegressor` and `DecisionTreeClassifier`
    
- Fully compatible with the project testing framework (**pytest**)
    

---

## Class API

### RandomForestRegressor

```
from rice_ml.random_forest  import RandomForestRegressor  

rf = RandomForestRegressor(     
	n_estimators=100,     
	max_depth=8,     
	min_samples_split=10,     
	max_features="sqrt",     
	random_state=42 
) 

rf.fit(X_train, y_train) 
y_pred = rf.predict(X_test)
```

### RandomForestClassifier

```
from rice_ml.random_forest  import RandomForestClassifier  

clf = RandomForestClassifier(     
	n_estimators=100,     
	max_depth=6,     
	random_state=42 
)  
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
```

---

## Notebook Overview — Real-World Regression Example

The example notebook demonstrates **Random Forest regression** on the  
**Seoul Bike Sharing Demand** dataset, where the goal is to predict hourly bike rental demand based on weather and temporal conditions.

The notebook focuses on illustrating **ensemble learning behavior**, rather than extensive feature engineering.

### Notebook includes:

1. Dataset loading with non-UTF-8 encoding handling
    
2. Feature selection using numeric weather and time variables
    
3. Train–test split for model evaluation
    
4. Training a **Decision Tree Regressor** as a baseline
    
5. Training a **Random Forest Regressor**
    
6. Model evaluation using **Mean Squared Error (MSE)**
    
7. Quantitative comparison between Decision Tree and Random Forest
    
8. Discussion of variance reduction and generalization improvements
    

---

## Key Results

- **Decision Tree** shows higher variance and less stable test performance
    
- **Random Forest** achieves **lower test MSE** and smoother predictions
    
- Ensemble averaging significantly improves generalization on unseen data
    

These results demonstrate how bagging reduces variance compared to a single high-capacity model.

---

## Conceptual Takeaways

- Decision Trees have **low bias but high variance**
    
- Random Forest reduces variance by:
    
    - training on bootstrap samples
        
    - introducing feature-level randomness
        
- Ensemble learning improves robustness without heavy preprocessing
    
- Random Forest is a strong baseline for many real-world regression tasks
    

---

## Unit Tests

Unit tests ensure correctness and stability of the implementation:

- Model instantiation
    
- Correct fitting behavior
    
- Output shape consistency
    
- Regression prediction averaging
    
- Classification majority voting
    
- Reasonable performance on simple synthetic datasets
    

Run tests with:

`pytest tests/test_random_forest.py -q`

---

## Summary

- Random Forest is a powerful ensemble method based on bagging
    
- It improves generalization by reducing variance
    
- The implementation is modular, testable, and reusable
    
- The example demonstrates practical benefits on real-world data
    

This module complements the Decision Tree implementation and illustrates the core idea of **ensemble learning in practice**.