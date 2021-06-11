# MART
A Model-Agnostic Rationale for Explaining Black-box Predictions: An Axiomatic Approach

We reestablish the axioms used in previous works and present a model-agnostic interpretation technique, Model-Agnostic RaTionale (MART), which measures the attribution of features by exploring all possible ranges of each feature. Reestablished axioms lead MART to avoid potentially misleading interpretations of previous attribution methods and to provide a theoretical justification. In short, MART is a feature attribution technique based on three axioms - Nullity, Implementation Invariance, and Continuity.

This directory contains implementations of MART framework the following UCI (tabular) datasets.
 - Breast Cancer Wisconsion (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
 - Biodegradation (https://www.kaggle.com/muhammetvarl/qsarbiodegradation)
 - Musk (https://www.kaggle.com/hashbanger/musk-dataset) 
 - Forest Fires (https://archive.ics.uci.edu/ml/datasets/forest+fires)
 - Mercedez-Benz (https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)

-- AUC values (of validation score decrease rate) for each dataset (The lower, the better). See run_tabular.py for details. --

|Method|wisc    |biodeg  |musk    |forest  |benz    |Average |
|------|--------|--------|--------|--------|--------|--------|
|MART  |**0.008495**|0.0799  |0.179952|0.067528|**0.210325**|**0.5462**|
|SHAP  |0.01299 |**0.048899**|**0.164965**|0.313326|0.688504|1.228684|
|LIME  |0.094136|0.199258|0.231901|**0.035452**|0.233401|0.794148|


Python Library Version:
- tensorflow == 1.10.0
- matplotlib == 3.1.3
- numpy == 1.14.5
- sklearn == 0.0
- pandas == 1.0.1
- seaborn == 0.10.0
- xgboost == 1.1.1
- shap == 0.37.0
- lime == 0.2.0.1
- gower == 0.0.5
- tqdm == 4.42.1
- scikit-image == 0.16.2
- scikit-learn == 0.23.2
- Pillow == 7.0.0
