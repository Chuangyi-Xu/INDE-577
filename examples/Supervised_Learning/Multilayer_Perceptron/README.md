# Multilayer Perceptron Module

This module implements a **Multi-Layer Perceptron (MLP)** neural network from  
scratch using NumPy. The implementation follows a clean, scikit-learn–style API  
and supports both classification and regression tasks. The module is fully  
integrated into the project with **unit tests** and a complete **end-to-end  
Jupyter Notebook example** based on a real-world marketing dataset.

---

## Features

- Feedforward neural network with one or more hidden layers
    
- Gradient-based optimization using backpropagation
    
- Support for mini-batch gradient descent
    
- Support for L2 regularization
    
- Reproducible training with `random_state`
    

**Configurable hyperparameters:**

- `hidden_layer_sizes`
    
- `activation`
    
- `learning_rate`
    
- `max_iter`
    
- `batch_size`
    
- `l2`
    
- `random_state`
    

**Training diagnostics:**

- Tracks training loss through `loss_curve_`
    

**Prediction interfaces:**

- `predict_proba()` — output class probability estimates
    
- `predict()` — output class labels
    
- `score()` — classification accuracy
    
- Fully compatible with the project testing framework (`pytest`)
    

---

## File Structure

`src/rice_ml/multilayer_perceptron.py 
`tests/test_multilayer_perceptron.py      
`examples/Multilayer_Perceptron.ipynb`

---

## Class API

```from rice_ml.multilayer_perceptron  import MLPClassifier  

mlp = MLPClassifier(     
     hidden_layer_sizes=(32, 16),
     activation="relu",     
     learning_rate=0.01,     
     max_iter=500,     
     batch_size=64,     
     l2=1e-4,     
     random_state=42 
)  

mlp.fit(X_train, y_train) 
y_pred = mlp.predict(X_test)
```

---

## Notebook Overview — Real-World Application

The example notebook demonstrates the application of a Multi-Layer Perceptron  
classifier on a **Bank Marketing dataset**, where the goal is to predict whether  
a customer will subscribe to a term deposit based on demographic, economic, and  
campaign-related features.

To ensure reproducibility and manageable repository size, a stratified subset  
of the original dataset is used while preserving the original class distribution.

---

### Notebook Includes

1. Dataset loading and exploratory analysis
    
2. Target encoding and feature preprocessing
    
3. One-hot encoding of categorical variables
    
4. Feature scaling for neural network training
    
5. Training a Multi-Layer Perceptron classifier
    
6. Visualization of training loss convergence
    
7. Model evaluation:
    
    - Accuracy
        
    - Confusion matrix
        
    - Classification report
        
8. Discussion of class imbalance and performance trade-offs
    

---

## Key Results

- Test accuracy of approximately **90%** on the held-out test set
    
- Stable and smooth convergence observed in the training loss curve
    
- Strong performance on the majority class with moderate recall on the minority  
    class, reflecting the inherent class imbalance of the dataset
    
- Demonstrates the effectiveness of nonlinear modeling compared to linear  
    baselines in complex, real-world classification tasks
    

---

## Unit Tests

Unit tests ensure the correctness and robustness of the implementation:

- Model initialization and parameter validation
    
- Forward and backward propagation consistency
    
- Training convergence on synthetic datasets
    
- `predict()` outputs valid class labels
    
- `predict_proba()` returns valid probability distributions
    
- Reproducibility under fixed random seeds
    

Run tests with:

`pytest tests/test_multilayer_perceptron.py -q`

---

## Summary

This module provides a clean, fully tested implementation of a Multi-Layer  
Perceptron from scratch, illustrating both the strengths and limitations of  
neural networks in applied machine learning settings. The accompanying example  
highlights the importance of preprocessing, evaluation metrics, and thoughtful  
model interpretation when applying MLPs to real-world data.