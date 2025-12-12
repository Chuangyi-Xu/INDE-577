## Linear Regression Module

This module implements a fully vectorized **multiple linear regression model** from scratch using NumPy.  
It follows a clean, scikit-learn–style API and is integrated into the project with **unit tests** and a complete **Jupyter Notebook example** based on the Auto MPG dataset.

---

### Features

- Gradient Descent optimization for minimizing Mean Squared Error (MSE)

- Support for:
  - `learning_rate`
  - `n_epochs`
  - `fit_intercept`
  - `tol`

- Tracks training progress through:
  - `losses_` — stores MSE at each iteration

- Returns learned model parameters:
  - `weights_`
  - `bias_`

- Includes:
  - `predict()` — continuous value prediction  
  - `score()` — computes R² coefficient of determination  

- Fully compatible with the project testing framework (`pytest`)

- Example notebook demonstrates:
  - Feature scaling for stable optimization
  - Loss curve visualization
  - Prediction vs. ground truth scatter plots
  - Residual diagnostics
  - Interpretation of learned coefficients
