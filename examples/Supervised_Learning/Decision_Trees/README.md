# Decision Trees Module

This module implements **Decision Tree classifiers and regressors from scratch**
using NumPy, following a clean, scikit-learn–style API.

The implementation supports both **classification** and **regression** tasks
and is fully integrated into the project with **unit tests** and a complete
**Jupyter Notebook example** based on a real-world dataset.

---

## Features

- Recursive binary tree construction (CART-style)
- Support for both:
  - **Classification** (Gini impurity / Entropy)
  - **Regression** (Mean Squared Error)
- Support for:
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `max_features`
  - `random_state`
- Built-in feature importance computation via impurity reduction
- Fully compatible with the project testing framework (`pytest`)

---

## Class API

```
from rice_ml.decision_trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

# Classification
clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Regression
reg = DecisionTreeRegressor(
    criterion="mse",
    max_depth=3,
    random_state=42
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

---

## Notebook Overview — Real-World Dataset

The example notebook demonstrates Decision Tree models using the  
**Wine Quality Dataset**, which contains physicochemical measurements  
of red wine samples and expert-rated quality scores.

Two learning tasks are explored:

- **Classification**:
    
    - Binary classification of wines as _good_ (quality ≥ 6) or _bad_
        
- **Regression**:
    
    - Predicting the wine quality score as a continuous variable
        

The dataset is well-suited for Decision Trees due to its non-linear  
relationships and fully numeric feature space.

---

## Notebook Includes

1. Data loading and preprocessing
    
2. Problem formulation for classification and regression
    
3. Decision Tree algorithm overview
    
4. Decision Tree classification experiments
    
5. Decision Tree regression experiments
    
6. Hyperparameter analysis:
    
    - Effect of `max_depth` on bias–variance tradeoff
        
7. Visualization of training vs test performance
    
8. Feature importance analysis
    
9. Discussion of piecewise constant predictions in regression
    
10. Summary and key takeaways
    

---

## Key Results

- **Classification**:
    
    - Training accuracy increases monotonically with tree depth
        
    - Test accuracy peaks at moderate depth and declines with overfitting
        
- **Regression**:
    
    - Training MSE decreases as tree depth increases
        
    - Test MSE exhibits a U-shaped curve, illustrating overfitting
        
    - Regression predictions are **piecewise constant**, reflecting  
        the averaging behavior within leaf nodes
        
- **Interpretability**:
    
    - Feature importance scores identify the most influential  
        physicochemical properties affecting wine quality
        

---

## Unit Tests

Unit tests ensure the correctness and robustness of the implementation:

- Model instantiation and parameter validation
    
- Correct tree construction for classification and regression
    
- Deterministic behavior under fixed `random_state`
    
- Valid prediction outputs and shapes
    
- Proper computation of feature importance
    

Run tests using:

`pytest tests/test_decision_trees.py -q`

---

## Summary

Decision Trees provide an intuitive and flexible approach to modeling  
complex, non-linear relationships.

While single trees are highly interpretable, they are prone to overfitting,  
motivating the use of ensemble methods such as **Random Forests**, which  
are explored in subsequent modules.