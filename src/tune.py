"""
src/tune.py
───────────
Hyperparameter optimisation using Optuna for the Telco Churn predictor.

Runs N trials of XGBoost or LightGBM, each logged as a child run
under a parent MLflow run. Best params are printed and saved to
models/best_params.json for use in src/train.py.

Usage
-----
    python src/tune.py --model xgb --trials 50
    python src/tune.py --model lgbm --trials 50
    python src/tune.py --model xgb --trials 100 --experiment tuning
"""
import os
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import build_preprocessor, run_pipeline

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress trial-by-trial output

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_PATH    = 'data/raw/telco_churn.csv'
MLFLOW_URI  = 'sqlite:///mlflow.db'
MODEL_NAME  = 'churn-model'

# scale_pos_weight for XGBoost — negative/positive ratio
# Computed from training set churn rate of 26.5%
SCALE_POS_WEIGHT = (1 - 0.265) / 0.265   # ≈ 2.77

def xgb_objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for XGBoost.

    Samples hyperparameters, trains a pipeline, and returns
    val PR-AUC (the metric we're optimising).

    Parameters suggested and their search ranges
    --------------------------------------------
    n_estimators     : 100–800  — more trees = potentially better but slower
    max_depth        : 3–8      — controls tree complexity
    learning_rate    : 0.01–0.3 — log scale because small values matter more
    subsample        : 0.6–1.0  — row subsampling
    colsample_bytree : 0.6–1.0  — feature subsampling per tree
    min_child_weight : 1–10     — minimum sum of instance weight in a leaf
    gamma            : 0–0.5    — minimum loss reduction to make a split
    reg_alpha        : 0–1.0    — L1 regularisation
    reg_lambda       : 0.5–2.0  — L2 regularisation
    """
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 800),
        'max_depth'        : trial.suggest_int('max_depth', 3, 8),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'gamma'            : trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 0.5, 2.0),
        'scale_pos_weight' : SCALE_POS_WEIGHT,
        'eval_metric'      : 'aucpr',
        'random_state'     : 42,
        'verbosity'        : 0,
    }

    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model',        XGBClassifier(**params)),
    ])

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    return average_precision_score(y_val, y_prob)

def lgbm_objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for LightGBM.

    Parameters suggested and their search ranges
    --------------------------------------------
    n_estimators      : 100–800
    num_leaves        : 20–100  — primary complexity control in LightGBM
    max_depth         : 3–8
    learning_rate     : 0.01–0.3 (log scale)
    feature_fraction  : 0.6–1.0
    bagging_fraction  : 0.6–1.0
    bagging_freq      : 1–10
    min_child_samples : 10–50   — overfitting control for small datasets
    reg_alpha         : 0–1.0
    reg_lambda        : 0–1.0
    """
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 800),
        'num_leaves'       : trial.suggest_int('num_leaves', 20, 100),
        'max_depth'        : trial.suggest_int('max_depth', 3, 8),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction' : trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq'     : trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 0.0, 1.0),
        'is_unbalance'     : True,
        'metric'           : 'average_precision',
        'random_state'     : 42,
        'verbose'          : -1,
    }

    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model',        LGBMClassifier(**params)),
    ])

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    return average_precision_score(y_val, y_prob)

def run_tuning(model_name: str, n_trials: int, experiment_name: str):
    """
    Run Optuna study for the specified model.
    Logs each trial as a child MLflow run under a parent run.
    Saves best params to models/best_params_{model_name}.json.

    Parameters
    ----------
    model_name      : 'xgb' or 'lgbm'
    n_trials        : number of Optuna trials to run
    experiment_name : MLflow experiment to log under
    """

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = run_pipeline(RAW_PATH)
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    # ── 2. Choose objective ───────────────────────────────────────────────────
    if model_name == 'xgb':
        objective_fn = xgb_objective
        display_name = 'XGBoost'
    elif model_name == 'lgbm':
        objective_fn = lgbm_objective
        display_name = 'LightGBM'
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'xgb' or 'lgbm'.")

    # ── 3. Configure MLflow ───────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    # ── 4. Run study inside a parent MLflow run ───────────────────────────────
    with mlflow.start_run(run_name=f'{display_name}_optuna_{n_trials}trials'):

        mlflow.log_param('model',    display_name)
        mlflow.log_param('n_trials', n_trials)
        mlflow.log_param('metric',   'val_pr_auc')

        trial_results = []

        def objective_with_logging(trial):
            """Wraps the objective to log each trial as a child MLflow run."""
            with mlflow.start_run(nested=True,
                                  run_name=f'trial_{trial.number}'):
                # Get params by running the objective up to the suggest calls
                # We need to actually call the objective to get the value
                val_pr_auc = objective_fn(
                    trial, X_train, y_train, X_val, y_val
                )

                # Log this trial's params and result
                for k, v in trial.params.items():
                    mlflow.log_param(k, v)
                mlflow.log_metric('val_pr_auc', val_pr_auc)

                # Also log ROC-AUC for reference
                pipeline = Pipeline([
                    ('preprocessor', build_preprocessor()),
                    ('model', XGBClassifier(**{
                        **trial.params,
                        'scale_pos_weight': SCALE_POS_WEIGHT,
                        'eval_metric': 'aucpr',
                        'random_state': 42,
                        'verbosity': 0,
                    }) if model_name == 'xgb' else LGBMClassifier(**{
                        **trial.params,
                        'is_unbalance': True,
                        'metric': 'average_precision',
                        'random_state': 42,
                        'verbose': -1,
                    }))
                ])
                pipeline.fit(X_train, y_train)
                y_prob = pipeline.predict_proba(X_val)[:, 1]
                val_roc_auc = roc_auc_score(y_val, y_prob)
                mlflow.log_metric('val_roc_auc', val_roc_auc)

                trial_results.append({
                    'trial'     : trial.number,
                    'val_pr_auc': val_pr_auc,
                    'val_roc_auc': val_roc_auc,
                })

                # Print progress every 10 trials
                if trial.number % 10 == 0:
                    print(f"  Trial {trial.number:3d} | "
                          f"PR-AUC={val_pr_auc:.4f} | "
                          f"Best so far={max(t['val_pr_auc'] for t in trial_results):.4f}")

            return val_pr_auc

        # ── 5. Create and run the study ───────────────────────────────────────
        print(f"\nRunning {n_trials} Optuna trials for {display_name}...")
        print(f"Optimising: val PR-AUC (higher is better)\n")

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        study.optimize(
            objective_with_logging,
            n_trials=n_trials,
            show_progress_bar=False,
        )

        # ── 6. Extract and log best results ───────────────────────────────────
        best_trial  = study.best_trial
        best_params = best_trial.params
        best_score  = best_trial.value

        print(f"\n{'='*50}")
        print(f"Optuna study complete — {n_trials} trials")
        print(f"{'='*50}")
        print(f"Best val PR-AUC : {best_score:.4f}")
        print(f"\nBest hyperparameters:")
        for k, v in best_params.items():
            print(f"  {k:<25} {v}")

        # Log best params and score to parent run
        mlflow.log_metric('best_val_pr_auc', best_score)
        for k, v in best_params.items():
            mlflow.log_param(f'best_{k}', v)

        # ── 7. Save best params to disk ───────────────────────────────────────
        os.makedirs('models', exist_ok=True)
        output = {
            'model'      : display_name,
            'best_params': best_params,
            'best_val_pr_auc': round(best_score, 4),
            'n_trials'   : n_trials,
        }
        outpath = f'models/best_params_{model_name}.json'
        with open(outpath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nBest params saved to {outpath}")
        mlflow.log_artifact(outpath)

        # ── 8. Plot optimisation history ──────────────────────────────────────
        plot_optimisation_history(study, trial_results, model_name, display_name)
        mlflow.log_artifact(f'reports/figures/optuna_history_{model_name}.png')
        mlflow.log_artifact(f'reports/figures/optuna_param_importance_{model_name}.png')

    return study, best_params


def plot_optimisation_history(study, trial_results, model_name, display_name):
    """Plot and save the optimisation history and parameter importance."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: trial scores over time with running best
    trial_nums  = [t['trial'] for t in trial_results]
    pr_aucs     = [t['val_pr_auc'] for t in trial_results]
    running_best = [max(pr_aucs[:i+1]) for i in range(len(pr_aucs))]

    axes[0].scatter(trial_nums, pr_aucs, alpha=0.4, s=20,
                    color='steelblue', label='Trial score')
    axes[0].plot(trial_nums, running_best, color='tomato',
                 linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial number')
    axes[0].set_ylabel('Val PR-AUC')
    axes[0].set_title(f'{display_name} — Optimisation History', fontweight='bold')
    axes[0].legend()

    # Right: parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        params_list  = list(importances.keys())[:8]   # top 8
        values_list  = [importances[p] for p in params_list]

        axes[1].barh(params_list[::-1], values_list[::-1],
                     color='steelblue', edgecolor='white')
        axes[1].set_xlabel('Importance')
        axes[1].set_title(f'{display_name} — Parameter Importance', fontweight='bold')
    except Exception:
        axes[1].text(0.5, 0.5, 'Not enough trials\nfor importance',
                     ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()
    path = f'reports/figures/optuna_history_{model_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save param importance separately too
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    try:
        importances = optuna.importance.get_param_importances(study)
        params_list  = list(importances.keys())[:8]
        values_list  = [importances[p] for p in params_list]
        ax2.barh(params_list[::-1], values_list[::-1],
                 color='steelblue', edgecolor='white')
        ax2.set_xlabel('Importance Score')
        ax2.set_title(f'{display_name} — Hyperparameter Importance', fontweight='bold')
    except Exception:
        pass
    plt.tight_layout()
    path2 = f'reports/figures/optuna_param_importance_{model_name}.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to reports/figures/")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning')
    parser.add_argument(
        '--model',
        type=str,
        default='xgb',
        choices=['xgb', 'lgbm'],
        help='Model to tune: xgb or lgbm'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of Optuna trials (default 50)'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='optuna-tuning',
        help='MLflow experiment name'
    )
    args = parser.parse_args()

    study, best_params = run_tuning(
        model_name      = args.model,
        n_trials        = args.trials,
        experiment_name = args.experiment,
    )

    print("\nDone. View runs at http://localhost:5000")

