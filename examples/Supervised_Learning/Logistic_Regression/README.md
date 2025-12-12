#  Logistic Regression Module

This module implements a fully vectorized **binary logistic regression classifier** from scratch using NumPy.  
It follows a clean, scikit-learn–style API and is integrated into the project with **unit tests** and a complete **Jupyter Notebook example** based on an industrial flotation plant dataset.

---

##  Features

- Gradient Descent optimization
    
- Support for:
    
    - `learning_rate`
        
    - `n_epochs`
        
    - `fit_intercept`
        
    - `random_state`
        
- Tracks training history through `losses_`
    
- Returns model parameters:
    
    - `weights_`
        
    - `bias_`
        
- Includes:
    
    - `predict_proba()` — output probability scores
        
    - `predict()` — binary classification predictions
        
- Fully compatible with the project testing framework (`pytest`)
    

---

## File Structure

`src/rice_ml/logistic_regression.py      # Core implementation tests/test_logistic_regression.py       # Unit tests examples/Logistic_Regression.ipynb      # End-to-end notebook`

---

##  Class API

`from rice_ml.logistic_regression import LogisticRegression  clf = LogisticRegression(     learning_rate=0.01,     n_epochs=1000,     fit_intercept=True )  clf.fit(X_train, y_train) y_pred = clf.predict(X_test)`

---

##  Notebook Overview — Industrial Application

The example notebook demonstrates logistic regression on **iron ore flotation process data**, where the goal is to predict whether the final concentrate contains **high silica content** (undesirable in steelmaking).

### Notebook includes:

1. **Data loading & numeric cleanup**
    
2. **Handling European-format numbers (comma decimals)**
    
3. **Missing value processing & variance filtering**
    
4. **Label engineering (`Quality_Label`)**
    
5. **Training a baseline logistic regression model**
    
6. **Feature scaling and improved training stability**
    
7. **Model evaluation:**
    
    - Accuracy
        
    - Confusion matrix
        
    - Classification report
        
    - ROC curve & AUC
        
8. **Feature interpretation:**
    
    - Coefficient ranking
        
    - Importance visualization
        

---

##  Key Results

- Baseline (unscaled) AUC: **0.507**
    
- After feature scaling, logistic regression improves to **AUC = 0.659**
    
- Most influential process variables:
    
    - **Amina Flow** (strongest positive contributor to high silica)
        
    - **Flotation Air Flow (Columns 01–03)** (negative contribution)
        
    - **Pulp pH** and **Starch Flow** (quality stabilizers)
        

---

##  Unit Tests

Unit tests ensure the correctness of the implementation:

- Model instantiation
    
- Training step updates weights
    
- `predict_proba()` returns valid probabilities
    
- `predict()` outputs valid binary labels
    
- The model learns a simple synthetic dataset perfectly
    

Run tests:

`pytest tests/test_logistic_regression.py -q`

---

##  Reference Notebook

The complete example is available at:

`examples/Logistic_Regression.ipynb`

It provides a full workflow from raw industrial dataset → feature engineering → model training → interpretability analysis.

---

##  Summary

This logistic regression module serves as:

- A **clean educational implementation** of gradient-based logistic regression
    
- A **reusable ML component** for the `rice_ml` package
    
- A **production-quality engineering example** applying ML to real industrial process data
    
- A **template** for building future algorithms in your model zoo (e.g., SVM, Random Forest, MLP)