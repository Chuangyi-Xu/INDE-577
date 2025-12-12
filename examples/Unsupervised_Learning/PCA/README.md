# Principal Component Analysis (PCA)

## 1. Introduction

Principal Component Analysis (PCA) is a classical **unsupervised learning**  
technique for **dimensionality reduction**.  
The main objective of PCA is to transform high-dimensional data into a  
lower-dimensional representation while preserving as much of the original  
variance as possible.

PCA is widely used in machine learning and data science for:

- Reducing dimensionality before applying supervised learning algorithms
    
- Removing redundancy caused by correlated features
    
- Data visualization in low-dimensional spaces
    
- Noise reduction and feature extraction
    

In this example, PCA is implemented **from scratch using NumPy** and applied  
to a real-world dataset from Kaggle.

---

## 2. Algorithm Overview

Given a dataset $\large X \in \mathbb{R}^{n \times d}$, PCA performs the following steps:

1. **Data Centering**  
    Subtract the empirical mean from each feature so that the data has zero mean.
    
2. **Covariance Matrix Computation**  
    Compute the covariance matrix of the centered data to capture feature  
    correlations.
    
3. **Eigen-Decomposition**  
    Perform eigen-decomposition of the covariance matrix to obtain:
    
    - Eigenvectors (principal directions)
        
    - Eigenvalues (variance explained by each direction)
        
4. **Component Selection and Projection**  
    Select the top k eigenvectors corresponding to the largest eigenvalues  
    and project the data onto this lower-dimensional subspace.
    

The resulting principal components are orthogonal and ordered by the amount  
of variance they explain.

---

## 3. Dataset Description

This example uses the **Breast Cancer Wisconsin (Diagnostic)** dataset obtained  
from Kaggle.

- Number of samples: 569
    
- Number of original features: 30 (continuous, real-valued)
    
- Target variable: Diagnosis (Malignant / Benign)
    

Although label information is available, PCA is applied in an **unsupervised**  
manner using only the numerical feature variables. The diagnosis labels are  
used solely for visualization and interpretation.

Prior to applying PCA:

- Non-numeric columns are excluded
    
- Zero-variance features are removed
    
- All features are standardized to zero mean and unit variance
    

---

## 4. Implementation Details

- PCA is implemented from scratch using **NumPy**, without relying on  
    `scikit-learn`.
    
- Feature standardization is performed manually to ensure numerical stability.
    
- Zero-variance features are removed to avoid division-by-zero issues.
    
- The explained variance ratio is computed to quantify the contribution of  
    each principal component.
    
- Scree plots and cumulative explained variance plots are used to guide the  
    selection of the number of components.
    

The implementation follows a **scikit-learn–style API**, supporting  
`fit`, `transform`, and `fit_transform` methods.

---

## 5. Experimental Results

Key observations from the PCA analysis include:

- The first principal component explains over 40% of the total variance.
    
- The first two principal components together explain more than 60% of the variance.
    
- Approximately 10 principal components are sufficient to preserve over 95%  
    of the total variance.
    
- A 2D PCA projection (PC1 vs PC2) reveals a clear separation trend between  
    malignant and benign samples, despite PCA being unsupervised.
    

These results indicate strong correlations among the original features and  
demonstrate the effectiveness of PCA for dimensionality reduction.

---

## 6. Visualization

The accompanying Jupyter notebook includes:

- Scree plots illustrating the explained variance of each component
    
- Cumulative explained variance plots for component selection
    
- A 2D PCA scatter plot colored by diagnosis labels for interpretability
    

These visualizations provide intuitive insights into the structure of the  
high-dimensional data after dimensionality reduction.

---

## 7. Summary and Key Takeaways

- PCA is a powerful unsupervised technique for reducing dimensionality and  
    uncovering latent structure in high-dimensional data.
    
- Implementing PCA from scratch highlights its linear algebra foundations.
    
- A small number of principal components can capture most of the variance in  
    real-world datasets with correlated features.
    
- PCA is particularly useful as a preprocessing step for downstream machine  
    learning models and for exploratory data analysis.