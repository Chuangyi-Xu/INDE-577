# Linear Regression Example

This example demonstrates how to apply a custom **Linear Regression** implementation to the Auto MPG dataset. 
The notebook walks through data preprocessing, model training with gradient descent, feature scaling, 
and evaluation using standard regression metrics and visualization tools.

---

## Overview

Linear Regression is a fundamental supervised learning algorithm used to model the relationship between 
a continuous target variable and one or more input features.  
In this example, we:

- Load and clean the Auto MPG dataset  
- Apply feature scaling for stable optimization  
- Train a custom linear regression model using gradient descent  
- Visualize loss convergence, predictions, and residuals  
- Interpret learned coefficients  

---

## Dataset

The dataset used in this example is the **Auto MPG dataset**, which includes information about vehicles such as:

- displacement  
- horsepower  
- weight  
- acceleration  
- model year  

The target variable is **mpg** (miles per gallon), representing fuel efficiency.

---

## Model Description

The Linear Regression model assumes a linear relationship between features and the target:

$$\large \hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

The parameters are optimized using **Gradient Descent** to minimize the Mean Squared Error (MSE):

$$\large \text{MSE} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2$$

Feature scaling is applied to ensure stable and efficient convergence.

---

## Training Procedure

The notebook includes:

1. Data loading and preprocessing  
2. Handling missing values  
3. Standardizing selected features using `StandardScaler`  
4. Training the Linear Regression model using gradient descent  
5. Plotting the training loss curve  
6. Evaluating performance with R² score  
7. Visualization of predictions and residuals  

