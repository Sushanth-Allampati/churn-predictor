"""
src/audit_runs.py
─────────────────
Prints a clean summary of all MLflow runs across all experiments.
Run this any time you want a quick overview of what's been trained.

Usage
-----
    python src/audit_runs.py
    python src/audit_runs.py --experiment model-comparison
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = 'sqlite:///mlflow.db'

def audit_runs(experiment_name: str = None):
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Get all experiments or filter by name
    all_experiments = client.search_experiments()

    if experiment_name:
        experiments = [e for e in all_experiments
                       if e.name == experiment_name]
        if not experiments:
            print(f"Experiment '{experiment_name}' not found.")
            print("Available experiments:",
                  [e.name for e in all_experiments])
            return
    else:
        experiments = all_experiments

    for exp in experiments:
        if exp.name == 'Default':
            continue

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=['metrics.val_roc_auc DESC']
        )

        print(f"\n{'='*65}")
        print(f"Experiment : {exp.name}")
        print(f"Runs found : {len(runs)}")
        print(f"{'='*65}")

        if not runs:
            print("  (no runs)")
            continue

        # Header
        print(f"  {'Model':<25} {'ROC-AUC':>8} {'PR-AUC':>8} "
              f"{'F1':>8} {'Recall':>8}  Run ID")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*8}")

        for run in runs:
            model  = run.data.params.get('model_name', 'unknown')[:24]
            roc    = run.data.metrics.get('val_roc_auc', 0)
            pr     = run.data.metrics.get('val_pr_auc',  0)
            f1     = run.data.metrics.get('val_f1',      0)
            recall = run.data.metrics.get('val_recall',  0)
            run_id = run.info.run_id[:8]

            print(f"  {model:<25} {roc:>8.4f} {pr:>8.4f} "
                  f"{f1:>8.4f} {recall:>8.4f}  {run_id}")

        # Best run summary
        best = runs[0]
        print(f"\n  Best run  : {best.info.run_id[:8]}")
        print(f"  Model     : {best.data.params.get('model_name', 'unknown')}")
        print(f"  Val ROC   : {best.data.metrics.get('val_roc_auc', 0):.4f}")
        print(f"  Val PR    : {best.data.metrics.get('val_pr_auc',  0):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default=None)
    args = parser.parse_args()
    audit_runs(args.experiment)