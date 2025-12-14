# Linear Regression Module

This module implements a fully vectorized **multiple linear regression model** from scratch using NumPy.  
It follows a clean, scikit-learn–style API and is integrated into the project with **unit tests** and a complete **Jupyter Notebook example** based on the Auto MPG dataset.

---

## Features

- Gradient Descent optimization for minimizing Mean Squared Error (MSE)
    
- Support for:
    
    - `learning_rate`
        
    - `n_epochs`
        
    - `fit_intercept`
        
    - `tol`
        
- Tracks training history through:
    
    - `losses_`
        
- Returns model parameters:
    
    - `weights_`
        
    - `bias_`
        
- Includes:
    
    - `predict()` — continuous value predictions
        
    - `score()` — R² coefficient of determination
        
- Fully compatible with the project testing framework (`pytest`)
    

---

## File Structure
```
src/rice_ml/linear_regression.py
        
tests/test_linear_regression.py
         
examples/Linear_Regression.ipynb        
```

---

## Class API

```
from rice_ml.linear_regression import LinearRegression  

model = LinearRegression(     
	learning_rate=0.01,     
	n_epochs=2000,     
	fit_intercept=True 
)  

model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
```

---

## Notebook Overview — Regression Application

The example notebook demonstrates linear regression on the **Auto MPG dataset**,  
where the goal is to predict **vehicle fuel efficiency (mpg)** based on engine and vehicle characteristics.

The notebook includes:

1. Data loading and numeric inspection
    
2. Handling missing values in raw automotive data
    
3. Feature selection for regression modeling
    
4. Train–test split for unbiased evaluation
    
5. Training a baseline linear regression model
    
6. Feature scaling for gradient descent stability
    
7. Model evaluation:
    
    - R² score
        
    - Prediction vs. true value visualization
        
    - Residual analysis
        
8. Feature interpretation:
    
    - Coefficient magnitude comparison
        
    - Directional impact of each feature
        

---

## Key Results

- Gradient descent converges smoothly after feature scaling
    
- The model achieves a reasonable **R² score** on both training and test sets
    
- Prediction vs. true value plots show strong linear alignment
    
- Residual plots indicate:
    
    - Approximate zero-mean residuals
        
    - Mild nonlinearity suggesting possible higher-order effects
        
- Most influential features on fuel efficiency:
    
    - **Weight** (strong negative impact on mpg)
        
    - **Displacement** (larger engines reduce fuel efficiency)
        
    - **Horsepower** (higher power correlates with lower mpg)
        

---

## Unit Tests

Unit tests ensure the correctness and stability of the implementation:

- Model instantiation and parameter initialization
    
- Gradient descent updates model weights correctly
    
- `predict()` outputs continuous numeric values
    
- `score()` returns valid R² values
    
- The model fits a simple synthetic linear dataset accurately
    

Run tests with:

`pytest tests/test_linear_regression.py -q`