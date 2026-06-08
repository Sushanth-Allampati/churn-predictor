"""
src/train.py
────────────
Model training script for the Telco Customer Churn predictor.

Trains a model inside a sklearn Pipeline (preprocessor + classifier),
logs everything to MLflow, and registers the best model in the
MLflow Model Registry.

Usage
-----
    python src/train.py                              # LR baseline
    python src/train.py --model xgb                 # XGBoost default params
    python src/train.py --model lgbm                # LightGBM default params
    python src/train.py --model xgb  --tuned        # XGBoost Optuna best params
    python src/train.py --model lgbm --tuned        # LightGBM Optuna best params
    python src/train.py --model lgbm --experiment my-exp

What is logged to MLflow for every run
---------------------------------------
    Params    : model name, all hyperparameters, random_state
    Metrics   : accuracy, ROC-AUC, PR-AUC, F1, precision, recall
                (both train_ and val_ prefixed)
    Artifacts : confusion_matrix.png, roc_curve.png,
                classification_report.txt, pipeline/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import warnings

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must be before pyplot import

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
from xgboost import XGBClassifier

from src.features import build_preprocessor, run_pipeline

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_PATH         = 'data/raw/telco_churn.csv'
MLFLOW_URI       = 'sqlite:///mlflow.db'
MODEL_NAME       = 'churn-model'

# Negative-to-positive class ratio for XGBoost scale_pos_weight.
# Computed from training set churn rate of 26.5%:
#   (1 - 0.265) / 0.265 ≈ 2.77
SCALE_POS_WEIGHT = (1 - 0.265) / 0.265


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(pipeline, X, y, split_name: str) -> dict:
    """
    Compute all evaluation metrics for one data split.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline
    X          : feature dataframe (un-transformed — pipeline handles transform)
    y          : true binary labels (0/1 Series or array)
    split_name : prefix for metric keys — 'train', 'val', or 'test'

    Returns
    -------
    dict mapping '{split_name}_{metric}' → float
    """
    y_pred      = pipeline.predict(X)
    y_pred_prob = pipeline.predict_proba(X)[:, 1]

    return {
        f'{split_name}_accuracy' : accuracy_score(y, y_pred),
        f'{split_name}_roc_auc'  : roc_auc_score(y, y_pred_prob),
        f'{split_name}_pr_auc'   : average_precision_score(y, y_pred_prob),
        f'{split_name}_f1'       : f1_score(y, y_pred, zero_division=0),
        f'{split_name}_precision': precision_score(y, y_pred, zero_division=0),
        f'{split_name}_recall'   : recall_score(y, y_pred, zero_division=0),
    }


# ── Artifact helpers ──────────────────────────────────────────────────────────

def save_confusion_matrix(pipeline, X_val, y_val, run_dir: str) -> str:
    """
    Save a confusion matrix PNG to run_dir and return the file path.
    Calls plt.close() to prevent memory leaks in long tuning runs.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_val, y_val,
        display_labels=['No Churn', 'Churned'],
        cmap='Blues', ax=ax,
    )
    ax.set_title('Confusion Matrix — Validation Set')
    path = os.path.join(run_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_roc_curve(pipeline, X_val, y_val, run_dir: str) -> str:
    """
    Save a ROC curve PNG to run_dir and return the file path.
    Overlays a random-classifier diagonal for reference.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(
        pipeline, X_val, y_val, ax=ax,
        name=pipeline.named_steps['model'].__class__.__name__,
    )
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
    ax.set_title('ROC Curve — Validation Set')
    ax.legend()
    path = os.path.join(run_dir, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_classification_report(pipeline, X_val, y_val, run_dir: str) -> str:
    """
    Save sklearn's full classification report as a .txt artifact.
    Includes per-class precision, recall, F1, and support.
    """
    y_pred  = pipeline.predict(X_val)
    report  = classification_report(
        y_val, y_pred,
        target_names=['No Churn', 'Churned'],
    )
    path = os.path.join(run_dir, 'classification_report.txt')
    with open(path, 'w') as f:
        f.write(report)
    return path


# ── Core training function ────────────────────────────────────────────────────

def train(model, params: dict, experiment_name: str):
    """
    Train a model inside a sklearn Pipeline, evaluate on train and val,
    log everything to MLflow, and register the pipeline in the Model Registry.

    The sklearn Pipeline ensures the preprocessor is always fit on training
    data only — calling pipeline.fit(X_train) fits both the ColumnTransformer
    and the classifier in one step with no leakage risk.

    Parameters
    ----------
    model           : unfitted sklearn-compatible classifier
    params          : hyperparameter dict logged to MLflow as params
    experiment_name : MLflow experiment to group this run under

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    metrics  : dict of all logged metric values
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading data from {RAW_PATH}...")
    X_train, X_val, X_test, y_train, y_val, y_test = run_pipeline(RAW_PATH)
    print("Data loaded.")

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('model',        model),
    ])

    # ── Configure MLflow ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nMLflow run started: {run_id[:8]}")

        # ── Train ─────────────────────────────────────────────────────────────
        print("Training...")
        pipeline.fit(X_train, y_train)
        print("Training complete.")

        # ── Evaluate ──────────────────────────────────────────────────────────
        train_metrics = evaluate_model(pipeline, X_train, y_train, 'train')
        val_metrics   = evaluate_model(pipeline, X_val,   y_val,   'val')
        all_metrics   = {**train_metrics, **val_metrics}

        # ── Log params ────────────────────────────────────────────────────────
        mlflow.log_param('model_name',   model.__class__.__name__)
        mlflow.log_param('random_state', 42)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # ── Log metrics ───────────────────────────────────────────────────────
        for k, v in all_metrics.items():
            mlflow.log_metric(k, v)

        # ── Print results table ───────────────────────────────────────────────
        _print_results_table(all_metrics)

        # ── Save and log artifacts ────────────────────────────────────────────
        run_dir = os.path.join('reports', 'mlflow_artifacts', run_id[:8])
        os.makedirs(run_dir, exist_ok=True)

        cm_path  = save_confusion_matrix(pipeline, X_val, y_val, run_dir)
        roc_path = save_roc_curve(pipeline, X_val, y_val, run_dir)
        rep_path = save_classification_report(pipeline, X_val, y_val, run_dir)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(rep_path)

        # ── Register model ────────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            pipeline,
            name='pipeline',
            registered_model_name=MODEL_NAME,
        )

        print(f"\nArtifacts saved to {run_dir}/")
        print(f"Model registered as '{MODEL_NAME}'.")
        print(f"View run: http://localhost:5000")

    return pipeline, all_metrics


def _print_results_table(metrics: dict):
    """Print a clean train/val comparison table to the terminal."""
    metric_keys = ['accuracy', 'roc_auc', 'pr_auc', 'f1', 'precision', 'recall']
    print(f"\n{'─'*45}")
    print(f"{'Metric':<25} {'Train':>8} {'Val':>8}")
    print(f"{'─'*45}")
    for m in metric_keys:
        tr = metrics.get(f'train_{m}', 0)
        vl = metrics.get(f'val_{m}',   0)
        print(f"  {m:<23} {tr:>8.4f} {vl:>8.4f}")
    print(f"{'─'*45}\n")


# ── Load Optuna best params ───────────────────────────────────────────────────

def load_best_params(model_name: str) -> dict:
    """
    Load Optuna best params from models/best_params_{model_name}.json.

    Parameters
    ----------
    model_name : 'xgb' or 'lgbm'

    Returns
    -------
    dict with keys 'model', 'best_params', 'best_val_pr_auc', 'n_trials'

    Raises
    ------
    FileNotFoundError if tune.py hasn't been run for this model yet.
    """
    path = f'models/best_params_{model_name}.json'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No tuned params at {path}. "
            f"Run: python src/tune.py --model {model_name}"
        )
    with open(path) as f:
        return json.load(f)


# ── Train from Optuna best params ─────────────────────────────────────────────

def train_from_best_params(model_name: str):
    """
    Load the best hyperparameters saved by Optuna and train a final model.
    Logs to the model-comparison experiment for direct comparison.

    Parameters
    ----------
    model_name : 'xgb' or 'lgbm'
    """
    config      = load_best_params(model_name)
    best_params = config['best_params']
    display     = config['model']

    print(f"\nTraining {display} with Optuna best params...")
    print(f"Expected val PR-AUC : {config['best_val_pr_auc']}")

    if model_name == 'xgb':
        model = XGBClassifier(
            **best_params,
            scale_pos_weight = SCALE_POS_WEIGHT,
            eval_metric      = 'aucpr',
            random_state     = 42,
            verbosity        = 0,
        )
    elif model_name == 'lgbm':
        model = LGBMClassifier(
            **best_params,
            is_unbalance = True,
            metric       = 'average_precision',
            random_state = 42,
            verbose      = -1,
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose 'xgb' or 'lgbm'.")

    return train(
        model           = model,
        params          = {**best_params, 'tuned_by': 'optuna'},
        experiment_name = 'model-comparison',
    )


# ── Default model configs ─────────────────────────────────────────────────────

def _build_lr():
    """Logistic Regression baseline — saga solver for sklearn 1.5+ compatibility."""
    params = {
        'C'           : 1.0,
        'max_iter'    : 1000,
        'solver'      : 'saga',
        'class_weight': 'balanced',
    }
    model = LogisticRegression(
        **params,
        random_state = 42,
    )
    return model, params


def _build_xgb():
    """XGBoost with sensible defaults and class imbalance correction."""
    params = {
        'n_estimators'    : 300,
        'max_depth'       : 4,
        'learning_rate'   : 0.05,
        'subsample'       : 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': round(SCALE_POS_WEIGHT, 4),
        'eval_metric'     : 'aucpr',
        'random_state'    : 42,
    }
    model = XGBClassifier(**params, verbosity=0)
    return model, params


def _build_lgbm():
    """LightGBM with sensible defaults and is_unbalance for class imbalance."""
    params = {
        'n_estimators'    : 300,
        'num_leaves'      : 31,
        'max_depth'       : -1,
        'learning_rate'   : 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq'    : 5,
        'min_child_samples': 20,
        'is_unbalance'    : True,
        'metric'          : 'average_precision',
        'random_state'    : 42,
    }
    model = LGBMClassifier(**params, verbose=-1)
    return model, params


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Ensure repo root is on the Python path when run as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(
        description='Train a churn prediction model and log to MLflow.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/train.py                              # Logistic Regression baseline
  python src/train.py --model xgb                 # XGBoost default params
  python src/train.py --model lgbm --tuned        # LightGBM Optuna best params
  python src/train.py --model xgb --experiment my-experiment
        """,
    )
    parser.add_argument(
        '--model',
        type=str,
        default='lr',
        choices=['lr', 'xgb', 'lgbm'],
        help='Model to train (default: lr)',
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='model-comparison',
        help='MLflow experiment name (default: model-comparison)',
    )
    parser.add_argument(
        '--tuned',
        action='store_true',
        help='Use Optuna best params from models/best_params_{model}.json',
    )
    args = parser.parse_args()

    # ── Dispatch to correct model ─────────────────────────────────────────────
    if args.tuned:
        if args.model == 'lr':
            parser.error("--tuned is not supported for Logistic Regression.")
        pipeline, metrics = train_from_best_params(args.model)

    elif args.model == 'lr':
        model, params = _build_lr()
        pipeline, metrics = train(model, params, args.experiment)

    elif args.model == 'xgb':
        model, params = _build_xgb()
        pipeline, metrics = train(model, params, args.experiment)

    elif args.model == 'lgbm':
        model, params = _build_lgbm()
        pipeline, metrics = train(model, params, args.experiment)

    print("\nDone. Open http://localhost:5000 to view the run.")