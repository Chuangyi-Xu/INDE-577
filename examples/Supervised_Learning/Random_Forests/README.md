# Random Forest

## Overview

Random Forest is an **ensemble learning method** that builds multiple decision trees and combines their predictions to improve generalization performance.  
By introducing randomness in both data sampling and feature selection, Random Forest reduces variance compared to a single decision tree while maintaining strong predictive power.

In this example, we demonstrate the Random Forest algorithm on a **real-world regression task** and compare its performance with a Decision Tree baseline.

---

## Algorithm Description

Random Forest is based on the principle of **bagging (bootstrap aggregating)**:

1. Multiple decision trees are trained on different **bootstrap samples** of the training data.
    
2. At each split, only a **random subset of features** is considered.
    
3. Predictions from all trees are combined:
    
    - **Regression:** average of predictions
        
    - **Classification:** majority vote
        

This approach reduces correlation among individual trees and leads to more stable and robust predictions.

---

## Dataset

We use the **Seoul Bike Sharing Demand** dataset, which contains hourly bike rental records along with weather-related and temporal features.

- **Task type:** Regression
    
- **Target variable:** `Rented Bike Count`
    
- **Features:** Weather conditions and time-related variables such as temperature, humidity, wind speed, and hour of the day
    

The dataset exhibits strong non-linear patterns, making it suitable for tree-based ensemble methods.

---

## Experimental Setup

### Models

Two models are trained and evaluated:

- **Decision Tree Regressor** (baseline)
    
- **Random Forest Regressor**
    

Both models are implemented from scratch in the `rice_ml` package using a scikit-learn–style API.

---

### Data Preprocessing

- Only numeric features are used
    
- Rows with missing values are removed
    
- No feature scaling is applied, as tree-based models are scale-invariant
    
- The dataset is split into training (80%) and testing (20%) sets
    

---

### Evaluation Metric

Model performance is evaluated using **Mean Squared Error (MSE)**:

$$\large \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Lower MSE indicates better predictive performance.

---

## Results

The Random Forest model consistently achieves **lower test MSE** than the single Decision Tree.

|Model|Test MSE|
|---|---|
|Decision decision Tree|Higher|
|Random Forest|Lower|

This demonstrates the effectiveness of ensemble learning in reducing variance and improving generalization.

---

## Discussion

The Decision Tree model can capture complex non-linear relationships but is sensitive to training data variations.  
Random Forest mitigates this issue by averaging multiple de-correlated trees, resulting in smoother predictions and better robustness to noise.

This example highlights the **bias–variance trade-off** and shows why Random Forest is often preferred over a single tree in practical applications.

---

## Files in This Directory

`Random_Forest/ ├── Random_Forest.ipynb └── README.md`

- `Random_Forest.ipynb`: Step-by-step implementation and comparison of Decision Tree and Random Forest models
    
- `README.md`: Overview and explanation of the Random Forest example
    

---

## Summary

- Random Forest is a powerful ensemble method for regression and classification
    
- It improves generalization by reducing variance through bootstrapping and feature randomness
    
- On the Seoul Bike Sharing dataset, Random Forest outperforms a single Decision Tree
    
- This example demonstrates the practical benefits of ensemble learning on real-world data