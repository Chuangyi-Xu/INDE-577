# Community Detection Module

This module implements a **graph-based community detection algorithm** using the **Girvan–Newman method**, developed from scratch and integrated into the project with a clean, scikit-learn–style API.

The implementation is designed for **unsupervised learning on network data** and is accompanied by **unit tests** and a complete **Jupyter Notebook example** using a real-world Kaggle social media dataset.

---

## Features

- Divisive community detection based on **edge betweenness centrality**
    
- Fully **unsupervised** (no labels required)
    
- Supports:
    
    - modularity-based model selection
        
    - reproducible results via `random_state`
        
- Tracks learned attributes:
    
    - `communities_` — list of detected communities
        
    - `labels_` — node-to-community mapping
        
    - `modularity_` — quality score of the selected partition
        
- Provides:
    
    - `fit()`
        
    - `predict()`
        
    - `fit_predict()`
        
- Handles **isolated nodes** gracefully (assigned label `-1`)
    
- Fully compatible with the project testing framework (`pytest`)
    

---
## Class API

```
from rice_ml.community_detection import GirvanNewmanCommunityDetection  

model = GirvanNewmanCommunityDetection(     
	keep_best_by_modularity=True,     
	random_state=42 
)  

model.fit(adjacency) 
labels = model.predict(adjacency)`
```

---

## Notebook Overview — Social Network Analysis

The example notebook demonstrates community detection on a **real-world Kaggle dataset of top Instagram influencers**.

Since the original dataset is tabular rather than graph-structured, the notebook illustrates how to:

- Transform tabular data into a network
    
- Apply graph-based unsupervised learning
    
- Interpret detected communities in a meaningful way
    

---

### Notebook Includes

1. Algorithm introduction (community detection & Girvan–Newman)
    
2. Data loading from Kaggle
    
3. Minimal data preprocessing and subsampling
    
4. Graph construction:
    
    - Nodes: influencers
        
    - Edges: shared country of origin
        
5. Community detection using the Girvan–Newman algorithm
    
6. Modularity-based community selection
    
7. Community visualization:
    
    - Community size distribution
        
    - Country composition per community
        
8. Interpretation of results and modeling assumptions
    

---

## Key Results

- Influencers are grouped into communities largely aligned with **geographic regions**
    
- Countries with many influencers form **large, dense communities**
    
- Countries with fewer influencers form **smaller, distinct communities**
    
- Influencers whose country appears only once are treated as **isolated nodes** and are not assigned to any community
    
- The results highlight how **graph construction choices directly influence detected communities**
    

---

## Unit Tests

Unit tests ensure the correctness and robustness of the implementation, including:

- Model initialization and fitting
    
- Correct handling of different input formats (adjacency dict, edge list, adjacency matrix)
    
- Proper behavior of `fit`, `predict`, and `fit_predict`
    
- Graceful handling of isolated nodes
    
- Reproducibility with fixed random seeds
    

Run tests:

`pytest tests/test_community_detection.py -q`

---

## Summary

This module demonstrates how classical graph-based algorithms can be applied to real-world data by thoughtfully transforming tabular datasets into network representations.  
The Girvan–Newman algorithm provides an interpretable and effective approach to uncovering community structure in social networks.