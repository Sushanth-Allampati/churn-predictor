"""
src/train.py
────────────
Model training script for the Telco Customer Churn predictor.

Trains a model inside a sklearn Pipeline (preprocessor + classifier),
logs everything to MLflow, and registers the best model in the
MLflow Model Registry.

Usage
-----
    python src/train.py                          # train with default config
    python src/train.py --experiment baseline    # named experiment
    python src/train.py --experiment xgboost     # swap the model

Logged to MLflow for every run
-------------------------------
    Params  : model name, all hyperparameters, random_state
    Metrics : accuracy, ROC-AUC, PR-AUC, F1, precision, recall
    Artifacts: trained pipeline (.pkl), confusion matrix plot,
               ROC curve plot, classification report (.txt)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.features import (
    build_preprocessor,
    run_pipeline,
)

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_PATH    = 'data/raw/telco_churn.csv'
MLFLOW_URI  = 'sqlite:///mlflow.db'          # local directory — mlflow ui reads from here
MODEL_NAME  = 'churn-model'     # name in the MLflow Model Registry

def evaluate_model(pipeline, X, y, split_name: str) -> dict:
    """
    Compute and return all evaluation metrics for one split.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline
    X          : feature dataframe
    y          : true labels (0/1)
    split_name : 'train', 'val', or 'test' — used for metric key prefixes

    Returns
    -------
    dict of metric_name → float
    """
    y_pred      = pipeline.predict(X)
    y_pred_prob = pipeline.predict_proba(X)[:, 1]

    metrics = {
        f'{split_name}_accuracy'  : accuracy_score(y, y_pred),
        f'{split_name}_roc_auc'   : roc_auc_score(y, y_pred_prob),
        f'{split_name}_pr_auc'    : average_precision_score(y, y_pred_prob),
        f'{split_name}_f1'        : f1_score(y, y_pred, zero_division=0),
        f'{split_name}_precision' : precision_score(y, y_pred, zero_division=0),
        f'{split_name}_recall'    : recall_score(y, y_pred, zero_division=0),
    }

    return metrics

def save_confusion_matrix(pipeline, X_val, y_val, run_dir: str):
    """Save confusion matrix plot as a PNG artifact."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_val, y_val,
        display_labels=['No Churn', 'Churned'],
        cmap='Blues', ax=ax
    )
    ax.set_title('Confusion Matrix — Validation Set')
    path = os.path.join(run_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_roc_curve(pipeline, X_val, y_val, run_dir: str):
    """Save ROC curve plot as a PNG artifact."""
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(
        pipeline, X_val, y_val, ax=ax,
        name=pipeline.named_steps['model'].__class__.__name__
    )
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
    ax.set_title('ROC Curve — Validation Set')
    ax.legend()
    path = os.path.join(run_dir, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_classification_report(pipeline, X_val, y_val, run_dir: str):
    """Save full classification report as a text artifact."""
    y_pred = pipeline.predict(X_val)
    report = classification_report(
        y_val, y_pred,
        target_names=['No Churn', 'Churned']
    )
    path = os.path.join(run_dir, 'classification_report.txt')
    with open(path, 'w') as f:
        f.write(report)
    return path

def train(model, params: dict, experiment_name: str):
    """
    Train a model inside a sklearn Pipeline, log everything to MLflow,
    and register the model in the MLflow Model Registry.

    Parameters
    ----------
    model           : unfitted sklearn-compatible classifier
    params          : hyperparameter dict — logged to MLflow
    experiment_name : MLflow experiment name (groups related runs)
    """
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data from {RAW_PATH}...")
    X_train, X_val, X_test, y_train, y_val, y_test = run_pipeline(RAW_PATH)
    print("Data loaded.")

    # ── 2. Build pipeline ─────────────────────────────────────────────────────
    # The preprocessor is always fitted on X_train inside pipeline.fit()
    # It is never fitted separately — sklearn Pipeline handles this correctly
    pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('model',        model),
    ])

    # ── 3. Configure MLflow ───────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    # ── 4. Start MLflow run ───────────────────────────────────────────────────
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nMLflow run started: {run_id}")

        # ── 5. Train ──────────────────────────────────────────────────────────
        print("Training...")
        pipeline.fit(X_train, y_train)
        print("Training complete.")

        # ── 6. Evaluate on train and val ──────────────────────────────────────
        train_metrics = evaluate_model(pipeline, X_train, y_train, 'train')
        val_metrics   = evaluate_model(pipeline, X_val,   y_val,   'val')
        all_metrics   = {**train_metrics, **val_metrics}

        # ── 7. Log params ─────────────────────────────────────────────────────
        mlflow.log_param('model_name',   model.__class__.__name__)
        mlflow.log_param('random_state', 42)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # ── 8. Log metrics ────────────────────────────────────────────────────
        for k, v in all_metrics.items():
            mlflow.log_metric(k, v)

        # ── 9. Print results to terminal ──────────────────────────────────────
        print(f"\n{'─'*45}")
        print(f"{'Metric':<25} {'Train':>8} {'Val':>8}")
        print(f"{'─'*45}")
        metrics_to_show = ['accuracy', 'roc_auc', 'pr_auc', 'f1',
                           'precision', 'recall']
        for m in metrics_to_show:
            tr = all_metrics[f'train_{m}']
            vl = all_metrics[f'val_{m}']
            print(f"  {m:<23} {tr:>8.4f} {vl:>8.4f}")
        print(f"{'─'*45}\n")

        # ── 10. Save and log artifacts ────────────────────────────────────────
        os.makedirs('reports/mlflow_artifacts', exist_ok=True)
        run_dir = f'reports/mlflow_artifacts/{run_id[:8]}'
        os.makedirs(run_dir, exist_ok=True)

        cm_path   = save_confusion_matrix(pipeline, X_val, y_val, run_dir)
        roc_path  = save_roc_curve(pipeline, X_val, y_val, run_dir)
        rep_path  = save_classification_report(pipeline, X_val, y_val, run_dir)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(rep_path)

        # ── 11. Log the model ─────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path='pipeline',
            registered_model_name=MODEL_NAME,
        )

        print(f"Artifacts saved to {run_dir}/")
        print(f"Model registered as '{MODEL_NAME}' in MLflow Model Registry.")
        print(f"\nView run at: http://localhost:5000/#/experiments/")

    return pipeline, all_metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train churn model')
    parser.add_argument(
        '--experiment',
        type=str,
        default='baseline',
        help='MLflow experiment name'
    )
    args = parser.parse_args()

    # ── Logistic Regression baseline ─────────────────────────────────────────
    # Logistic Regression is the baseline — not because it's expected to win,
    # but because it sets a floor that tree models must beat to be justified.
    # C=1.0 is the default regularisation strength — no tuning at this stage.

    lr_params = {
        'C'          : 1.0,
        'max_iter'   : 1000,
        'solver'     : 'lbfgs',
        'class_weight': 'balanced',  # handles 73/27 imbalance automatically
    }

    model = LogisticRegression(
        C            = lr_params['C'],
        max_iter     = lr_params['max_iter'],
        solver       = lr_params['solver'],
        class_weight = lr_params['class_weight'],
        random_state = 42,
    )

    pipeline, metrics = train(
        model=model,
        params=lr_params,
        experiment_name=args.experiment,
    )

    print("Done. Open http://localhost:5000 to view the run.")
