# K-Nearest Neighbors (KNN)

## Overview

The k-Nearest Neighbors (KNN) algorithm is a non-parametric, instance-based  
learning method commonly used for classification problems. Unlike parametric  
models that learn an explicit decision function during training, KNN makes  
predictions by comparing a new observation to the most similar samples in the  
training set based on a distance metric.

In this example, we implement KNN from scratch using NumPy and apply it to a  
real-world dataset from Kaggle to study the effect of hyperparameters such as  
the number of neighbors and distance metrics on classification performance.

---

## Algorithm Description

Given a training dataset

$$\large \{(\mathbf{x}_i, y_i)\}_{i=1}^n$$

the KNN classifier predicts the label of a new observation $\large \mathbf{x}$ by:

1. Computing the distance between $\large \mathbf{x}$ and all training samples  
    using a predefined distance metric.
    
2. Selecting the k nearest neighbors with the smallest distances.
    
3. Assigning the class label by majority voting among these neighbors.
    

Because KNN does not assume any functional form of the decision boundary, it is  
capable of modeling complex, nonlinear relationships in the data.

---

## Distance Metrics

In this implementation, the following distance metrics are supported:

- **Euclidean distance**
    

$$\large d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{p}(x_j - x'_j)^2}$$

- **Manhattan distance**
    

$$\large d(\mathbf{x}, \mathbf{x}') = \sum_{j=1}^{p} |x_j - x'_j|$$

The choice of distance metric can significantly affect model performance,  
especially when features are measured on different scales.

---

## Dataset

The dataset used in this example is the **Breast Cancer Wisconsin (Diagnostic)**  
dataset obtained from Kaggle. It contains 569 samples with 30 numerical features  
describing characteristics of cell nuclei computed from digitized images of  
fine needle aspirates (FNA) of breast masses.

The target variable indicates whether a tumor is malignant (M) or benign (B).

---

## Data Preprocessing

Since KNN is a distance-based algorithm, feature scaling is essential.  
All features are standardized using Z-score normalization to ensure that each  
feature contributes equally to the distance computation.

The dataset is split into training and testing sets to evaluate the model’s  
generalization performance.

---

## Model Training and Evaluation

The KNN classifier is trained using the standardized training data and evaluated  
on a held-out test set. Classification accuracy is used as the evaluation metric.

The results show that KNN achieves high accuracy on this dataset, indicating  
that local similarity patterns in the feature space are informative for  
classification.

---

## Hyperparameter Analysis

To study the impact of the number of neighbors k, the model is evaluated  
across a range of odd values of k. The results demonstrate the classical  
bias–variance trade-off:

- Small values of k lead to high variance and sensitivity to noise.
    
- Large values of k produce smoother decision boundaries with higher bias.
    
- An intermediate value of k achieves the best classification performance.
    

---

## Files in This Directory

- `KNN.ipynb`  
    Jupyter notebook containing algorithm explanation, data preprocessing,  
    model training, and hyperparameter analysis.
    
- `data/`  
    Directory containing the Kaggle dataset used in this example.
    

---

## Summary

This example demonstrates a complete end-to-end implementation of the  
k-Nearest Neighbors algorithm, including algorithm design, preprocessing,  
model evaluation, and hyperparameter analysis. It highlights the strengths  
and limitations of KNN and illustrates how distance-based learning methods  
can be applied to real-world classification problems.