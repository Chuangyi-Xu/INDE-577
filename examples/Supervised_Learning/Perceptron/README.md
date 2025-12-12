## Perceptron

The **Perceptron** is one of the earliest supervised learning algorithms for binary classification.  
Proposed by _Frank Rosenblatt_ in 1958, it models a simplified biological neuron and forms the basis of many modern linear classifiers and neural network architectures.

This implementation follows a **scikit-learn–style API** and is built entirely from scratch without using machine learning libraries.

---

###  **Algorithm Overview**

Given an input vector $\large \mathbf{x}$, the perceptron computes a linear prediction:

$$\large f(\mathbf{x}) = \begin{cases} 1, & \text{if } \mathbf{w} \cdot \mathbf{x} + b > 0 \\ 0, & \text{otherwise} \end{cases}$$

where

- $\large \mathbf{w}$ = weight vector
    
- $\large b$ = bias term
    
- $\large \mathbf{w} \cdot \mathbf{x}$ = dot product
    

The learning algorithm iteratively corrects misclassified samples using the update rule:

$$\large \mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y})\mathbf{x}$$$$\large b \leftarrow b + \eta (y - \hat{y})$$

The perceptron converges **only when data is linearly separable**.

---

###  **Source Code**

Implementation:

`src/rice_ml/perceptron.py`

Key API:

`clf = Perceptron(max_iter=1000, lr=1.0) clf.fit(X, y) y_pred = clf.predict(X) score = clf.score(X, y)`

Features:

- Handles labels in `{0, 1}` or `{-1, 1}`
    
- Random weight initialization with reproducibility (`random_state`)
    
- Binary classification using a linear decision boundary
    
- Compatible with unit tests and example notebooks
    

---

###  **Unit Tests**

Tests for correctness:

`tests/test_perceptron.py`

The tests ensure:

- Correct fitting on a simple AND gate dataset
    
- Predict accuracy matches expected output
    
- Score method returns valid values between 0 and 1
    
- API consistency with scikit-learn conventions
    

To run tests:

`pytest tests/test_perceptron.py -q`

---

###  **Example Notebook**

Notebook demonstrating data preparation, training, visualization, and interpretation:

`examples/Supervised_Learning/Perceptron/Perceptron.ipynb`

Highlights:

- Uses Iris dataset (Setosa vs Versicolor)
    
- Visualizes decision boundary
    
- Achieves perfect accuracy on linearly separable subset
    
- Includes algorithm explanation and interpretation
    

---

###  **Advantages**

- Simple and efficient
    
- Easy to implement from scratch
    
- Works well for linearly separable datasets
    
- Computationally light
    

---

###  **Limitations**

- Cannot model non-linear decision boundaries
    
- Fails to converge on non-linearly separable data
    
- Produces hard binary predictions (no probability scores)
    

---

###  **Summary**

The perceptron implemented in this project provides:

- A clean and modular linear classifier
    
- Full compatibility with unit tests and examples
    
- A solid foundation for understanding more advanced models such as Logistic Regression, SVMs, and neural networks