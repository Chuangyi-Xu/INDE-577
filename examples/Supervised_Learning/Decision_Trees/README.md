# Decision Trees

## Overview

Decision Trees are non-parametric, tree-based machine learning models used
for both **classification** and **regression** tasks.

Unlike linear models, Decision Trees do not assume any specific functional
relationship between input features and the target variable. Instead, they
learn a set of hierarchical decision rules that recursively partition the
feature space into smaller, more homogeneous regions.

In this example, we implement and analyze Decision Tree models using a
real-world dataset to illustrate their behavior, strengths, and limitations.

---

## Algorithm Description

A Decision Tree consists of a sequence of binary splits applied to the input
features.

At each internal node, the algorithm selects:
- a feature $\large x_j$ 
- a threshold $\large t$ 

and applies the decision rule:

$\large x_j \le t$

This process continues recursively until a stopping criterion is met,
such as a maximum tree depth or a minimum number of samples per leaf.

### Classification Trees

For classification tasks, Decision Trees aim to increase node purity.
Common impurity measures include:

- **Gini Impurity**

$$\large G = 1 - \sum_{k=1}^{K} p_k^2$$

- **Entropy**

$$\large H = - \sum_{k=1}^{K} p_k \log_2(p_k)$$

where $\large p_k$ denotes the proportion of samples belonging to class $\large k$ 
within a node.

The optimal split is chosen to maximize impurity reduction.

### Regression Trees

For regression tasks, Decision Trees aim to minimize the variability of
target values within each node.

The most commonly used criterion is **Mean Squared Error (MSE)**:

$$\large \mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$$
At each leaf node, the predicted value is the **mean** of the target values
of the samples contained in that leaf.

As a result, Decision Tree regression produces **piecewise constant**
predictions rather than smooth curves.

---

## Dataset

The experiments in this notebook use the **Wine Quality Dataset**, which
contains physicochemical measurements of red wine samples and a quality
score assigned by human experts.

- All features are continuous numeric variables
- The dataset exhibits strong non-linear relationships
- It is well-suited for demonstrating both classification and regression
  behavior of Decision Trees

Two learning tasks are considered:
- **Classification**: Predict whether a wine is *good* (quality ≥ 6) or *bad*
- **Regression**: Predict the wine quality score directly

---

## Experiments and Results

The notebook investigates the behavior of Decision Trees under different
model complexities.

### Effect of Tree Depth

- Increasing tree depth reduces training error monotonically.
- Test performance improves up to a certain depth and then degrades,
  demonstrating overfitting.
- This behavior illustrates the **bias–variance tradeoff** inherent in
  Decision Tree models.

Both classification accuracy and regression mean squared error are analyzed
as functions of tree depth.

### Feature Importance

Decision Trees provide intrinsic measures of feature importance based on
impurity reduction.

Features with higher importance values contribute more significantly to
the model's decision-making process, offering valuable interpretability.

---

## Key Takeaways

- Decision Trees are flexible models capable of capturing complex,
  non-linear relationships.
- Tree depth is a critical hyperparameter that must be carefully tuned.
- Regression trees produce piecewise constant predictions due to averaging
  within leaf nodes.
- While interpretable, single Decision Trees can be unstable and prone to
  overfitting.

Decision Trees serve as a strong baseline model and form the foundation
for ensemble methods such as Random Forests and Gradient Boosting.

---

## Files in This Directory

- `Decision_Trees.ipynb`  
  Jupyter notebook containing algorithm explanation, experiments, and
  visual analysis for Decision Tree classification and regression.

- `README.md`  
  This file.
