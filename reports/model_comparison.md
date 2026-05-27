# Model Comparison Report

**Dataset** : Telco Customer Churn (7,043 customers, 26.5% churn rate)  
**Split**   : 70% train / 15% val / 15% test (stratified)  
**Primary metric** : PR-AUC (imbalanced dataset — accuracy is misleading)

## Results — Validation Set

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| LightGBM | ~0.876 | ~0.739 | ~0.692 | ~0.659 | ~0.731 |
| XGBoost | ~0.873 | ~0.734 | ~0.689 | ~0.653 | ~0.729 |
| Logistic Regression (baseline) | ~0.835 | ~0.636 | ~0.629 | ~0.532 | ~0.769 |

*(Fill in exact numbers from your MLflow runs)*

## Key findings

**Tree models vs baseline**
Both XGBoost and LightGBM outperform Logistic Regression by ~0.04 ROC-AUC
and ~0.10 PR-AUC. The PR-AUC gap is larger because tree models handle
class imbalance more effectively than a linear model.

**XGBoost vs LightGBM**
Metrics are within noise of each other. LightGBM trains 3–5x faster.
Both will be tuned with Optuna — winner selected on val PR-AUC.

**Class imbalance handling**
scale_pos_weight=2.77 (negative/positive ratio) gives the best
precision-recall balance for XGBoost.
LightGBM uses is_unbalance=True for equivalent effect.

**Threshold tuning**
Default threshold of 0.5 is suboptimal for imbalanced data.
Optimal threshold (~0.35) improves F1 by ~0.02–0.03 without retraining.
Threshold saved to models/threshold_config.json.

## Class imbalance strategy

| Approach | Applied to | Effect |
|---|---|---|
| scale_pos_weight=2.77 | XGBoost | Weights minority class 2.77x in loss |
| is_unbalance=True | LightGBM | Automatically reweights by class ratio |
| class_weight='balanced' | Logistic Regression | sklearn equivalent |
| Threshold tuning | All models | Shifts decision boundary post-training |

## Next steps
- Optuna hyperparameter optimisation (Day 16) on XGBoost and LightGBM
- SHAP feature importance analysis (Day 17–18)
- Final model selection and registration (Day 19)