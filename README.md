# Credit Card Default Prediction

Statistical Learning project developed at the University of Haifa by **Andrey Makushev** and **Itay Pizanty**.

## Overview

This project analyzes the **UCI Default of Credit Card Clients** dataset and compares multiple classification approaches for predicting whether a credit card customer will default.

The workflow includes exploratory data analysis, feature engineering, model tuning, dimensionality reduction, clustering, permutation feature importance, and comparative model evaluation.

## Dataset

The notebook loads the dataset directly through `ucimlrepo` using UCI dataset ID `350`.

Main feature groups:

- Credit limit and demographic attributes
- Six months of repayment status
- Six months of bill amounts
- Six months of previous payments
- Engineered features:
  - `Debt_Ratio`
  - `Avg_Payment`

**Target:** `default`
- `1` — customer defaulted
- `0` — customer did not default

## Workflow

1. Data loading and cleaning
2. Exploratory data analysis
3. Feature engineering and preprocessing
4. Train/test split and 5-fold cross-validation setup
5. Model training and hyperparameter tuning
6. Overfitting and bias-variance diagnostics
7. PCA experiments
8. K-Means clustering experiments
9. Permutation Feature Importance (PFI)
10. Final model comparison

## Models

The project evaluates:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- Deep Neural Network (DNN)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)

Hyperparameters are tuned using `GridSearchCV` and `RandomizedSearchCV` where applicable.

## Evaluation

Models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Training time

The notebook also includes train/test comparisons, cross-validation results, learning curves, and ROC curves.

## Results

In the final comparison, **XGBoost with Permutation Feature Importance** achieved the highest reported ROC-AUC:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost (PFI) | 0.8183 | 0.6695 | 0.3529 | 0.4622 | **0.7789** |
| XGBoost | 0.8181 | 0.6715 | 0.3484 | 0.4588 | 0.7785 |
| DNN | 0.8173 | 0.6486 | 0.3801 | 0.4793 | 0.7708 |
| Random Forest (PFI) | **0.8203** | 0.6828 | 0.3507 | 0.4634 | 0.7667 |
| Random Forest | 0.8196 | 0.6799 | 0.3492 | 0.4614 | 0.7651 |

Additional experiments showed mixed effects from PCA and K-Means clustering, while PFI-based reduced feature sets produced similar or slightly improved reported performance for several models.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- TensorFlow / Keras
- SciKeras
- UCI ML Repository (`ucimlrepo`)

## Running the Project

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
credit_card_default_prediction.ipynb
```

The dataset is downloaded directly by the notebook, so no separate data file is required.

## Authors

**Andrey Makushev**  
**Itay Pizanty**

University of Haifa — Statistical Learning project.
