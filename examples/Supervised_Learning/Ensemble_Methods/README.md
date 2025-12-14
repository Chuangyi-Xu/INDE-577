# Ensemble Methods Module

This module demonstrates a simple **ensemble learning strategy** using a
**hard voting classifier**, built on top of multiple base models implemented
in this project.

Rather than introducing a new learning algorithm, the ensemble method
illustrates how existing classifiers can be combined to produce more stable
and robust predictions through majority voting.

The implementation follows a clean, scikit-learn–style API and is fully
integrated with the project’s testing and example framework.

---

## Features

- Hard voting ensemble for binary classification
- Combines multiple base classifiers with identical interfaces
- Compatible with any model implementing:
  - `fit(X, y)`
  - `predict(X)`
- Lightweight meta-model (no additional training parameters)
- Fully tested with `pytest`

---

## Class API

```
from rice_ml.ensemble_methods import VotingClassifier
from rice_ml.knn import KNNClassifier
from rice_ml.logistic_regression import LogisticRegression

knn = KNNClassifier(k=5)
logreg = LogisticRegression(
    learning_rate=0.01,
    n_epochs=1000,
    random_state=42
)

ensemble = VotingClassifier(models=[knn, logreg])
ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
```

---
## Notebook Overview — Model Combination Example

The example notebook demonstrates ensemble learning on a standard  
binary classification task.

The notebook includes:

1. Data loading and train–test split
    
2. Training individual base classifiers
    
3. Evaluating standalone model performance
    
4. Constructing a hard voting ensemble
    
5. Comparing ensemble performance against individual models
    
6. Discussion of ensemble robustness and stability
    

The focus is on **conceptual clarity** rather than hyperparameter tuning or  
model optimization.

---

## Key Takeaways

- Ensemble learning improves robustness by aggregating multiple models
    
- Hard voting is a simple yet effective ensemble strategy
    
- Ensemble methods can be implemented without modifying base model internals
    
- The design cleanly separates base learners from the meta-model logic
    

---

## Unit Tests

Unit tests verify the correctness and stability of the ensemble implementation:

- Successful model instantiation
    
- Proper delegation of `fit()` to all base models
    
- Correct output shape from `predict()`
    
- Compatibility with existing project classifiers
    

Run tests with:

`pytest tests/test_ensemble_methods.py -q`

---

## Notes

This ensemble implementation is intentionally minimal and educational,  
designed to illustrate the **core idea of ensemble learning** within the  
context of a from-scratch machine learning library.