# Regression Tree 

This module implements a **CART-style regression tree** from scratch using NumPy.  
The implementation follows a clean, **scikit-learn–style API** and is fully integrated into the project with **unit tests** and a complete **Jupyter Notebook example** based on a real-world housing dataset.

---

## Features

- CART regression tree using **Mean Squared Error (MSE)** as the splitting criterion
    
- Recursive binary partitioning of the feature space
    
- Support for key hyperparameters:
    
    - `max_depth`
        
    - `min_samples_split`
        
    - `min_samples_leaf`
        
    - `max_features`
        
    - `random_state`
        
- Produces **piecewise constant predictions**
    
- Computes feature importance based on impurity reduction
    
- Fully compatible with the project testing framework (`pytest`)
    

---

## File Structure

`src/rice_ml/regression_trees.py`        

`tests/test_regression_trees.py`         

`examples/Regression_Tree.ipynb`            

---

## Class API

```from rice_ml.regression_trees import RegressionTreeRegressor  
model = RegressionTreeRegressor(     
    max_depth=3,     
    min_samples_leaf=20,     
    random_state=42 
    )  
    
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)`
```
---

## Notebook Overview — Housing Price Regression

The example notebook demonstrates the application of a regression tree to the **California Housing dataset**, where the goal is to predict **median house value** based on socioeconomic and geographic features.

To emphasize interpretability, the notebook focuses on a **one-dimensional regression setting**, using median income as the explanatory variable.

### Notebook includes:

1. Dataset introduction and problem formulation
    
2. Data loading from `sklearn.datasets.fetch_california_housing`
    
3. Feature and target selection
    
4. Train–test split
    
5. Training a regression tree model from scratch
    
6. Model evaluation using Mean Squared Error (MSE)
    
7. Visualization of:
    
    - Original data distribution
        
    - Piecewise constant regression tree predictions
        
8. Interpretation of tree behavior and split structure
    

---

## Key Results

- The regression tree successfully captures the **nonlinear relationship** between median income and house value
    
- Learned predictions exhibit a clear **stepwise structure**, corresponding to leaf node averages
    
- Hyperparameters such as tree depth and minimum leaf size strongly influence the bias–variance trade-off
    
- Visualization highlights the interpretability advantage of tree-based models over linear regression
    

---

## Unit Tests

Unit tests ensure the correctness and robustness of the implementation:

- Model instantiation and parameter validation
    
- Correct behavior of `fit()` and `predict()`
    
- Proper enforcement of stopping conditions
    
- Shape and type validation of predictions
    
- Feature importance computation
    
- Error handling for invalid inputs
    

Run tests with:

`pytest tests/test_regression_trees.py -q`

---

## Summary

This regression tree implementation provides an interpretable, nonparametric approach to regression problems.  
While individual trees may suffer from high variance, they form the foundational building blocks for powerful ensemble methods such as **Random Forests** and **Gradient Boosting**, which are explored in subsequent modules.

---

## Notes

- This implementation is designed for **educational purposes**, emphasizing clarity and correctness over performance optimization.
    
- The notebook example is fully reproducible and independent of external data files.