"""
Model Performance Monitoring

This module tracks model performance metrics over time to detect degradation.
Key metrics monitored:
- AUC-ROC
- KS Statistic
- Gini Coefficient
- Precision, Recall, F1
- Expected Cost
"""

import logging
from datetime import datetime
from io import BytesIO
from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.io import get_s3_client, load_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configs

cfg = load_config()

with open("configs/costs.yaml") as f:
    costs_cfg = yaml.safe_load(f)

MINIO = cfg["minio"]
BUCKET = MINIO["bucket"]

s3 = get_s3_client(cfg)

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "auc": {"min": 0.65, "warning": 0.70},
    "ks": {"min": 0.20, "warning": 0.25},
    "gini": {"min": 0.30, "warning": 0.40},
}

COST_FN = costs_cfg["costs"]["false_negative"]
COST_FP = costs_cfg["costs"]["false_positive"]


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Gini coefficient (2*AUC - 1)."""
    auc = roc_auc_score(y_true, y_prob)
    return float(2 * auc - 1)


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate expected cost based on confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(COST_FN * fn + COST_FP * fp)


def calculate_lift(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Calculate lift table for model evaluation.

    Lift shows how much better the model is at identifying defaults
    compared to random selection, across different score deciles.
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decile"] = pd.qcut(df["y_prob"], q=n_bins, labels=False, duplicates="drop")

    # Group by decile (higher decile = higher predicted risk)
    lift_table = (
        df.groupby("decile")
        .agg(
            n_total=("y_true", "count"),
            n_bads=("y_true", "sum"),
            avg_prob=("y_prob", "mean"),
        )
        .reset_index()
    )

    lift_table["bad_rate"] = lift_table["n_bads"] / lift_table["n_total"]
    overall_bad_rate = y_true.mean()
    lift_table["lift"] = lift_table["bad_rate"] / overall_bad_rate

    # Cumulative metrics (from highest risk decile)
    lift_table = lift_table.sort_values("decile", ascending=False)
    lift_table["cum_bads"] = lift_table["n_bads"].cumsum()
    lift_table["cum_total"] = lift_table["n_total"].cumsum()
    lift_table["cum_bad_rate"] = lift_table["cum_bads"] / lift_table["cum_total"]
    lift_table["cum_capture"] = lift_table["cum_bads"] / y_true.sum()

    return lift_table.sort_values("decile")


def calculate_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """Calculate all performance metrics."""

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "gini": gini_coefficient(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "expected_cost": expected_cost(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
        "n_samples": len(y_true),
        "default_rate": float(y_true.mean()),
    }

    return metrics


def evaluate_performance_status(metrics: Dict) -> Dict:
    """
    Evaluate if metrics meet thresholds.

    Returns status for each key metric.
    """
    status = {}

    for metric_name, thresholds in PERFORMANCE_THRESHOLDS.items():
        value = metrics.get(metric_name, 0)

        if value < thresholds["min"]:
            status[metric_name] = "CRITICAL"
        elif value < thresholds["warning"]:
            status[metric_name] = "WARNING"
        else:
            status[metric_name] = "OK"

    # Overall status
    if any(s == "CRITICAL" for s in status.values()):
        status["overall"] = "CRITICAL"
    elif any(s == "WARNING" for s in status.values()):
        status["overall"] = "WARNING"
    else:
        status["overall"] = "OK"

    return status


def load_csv(key: str) -> pd.DataFrame:
    """Load CSV from MinIO."""
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))


def upload_csv(df: pd.DataFrame, key: str) -> None:
    """Upload DataFrame as CSV to MinIO."""
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())


def run_performance_monitoring(
    model_uri: Optional[str] = None,
    data_key: Optional[str] = None,
    target_col: str = "default_90p_12m",
    threshold: float = 0.5,
    output_key: Optional[str] = None,
) -> Dict:
    """
    Run performance monitoring for a deployed model.

    Run performance monitoring.
    """
    final_output_key = output_key or (MINIO["paths"]["monitoring"] + "mon2_performance_report.csv")
    data_key = data_key or (MINIO["paths"]["processed"] + "test.csv")

    logger.info(f"Loading evaluation data: {data_key}")
    df = load_csv(data_key)  # type: ignore

    y_true = df[target_col].values
    X = df.drop(columns=[target_col])

    # Load model from MLflow if URI provided
    if model_uri:
        logger.info(f"Loading model from MLflow: {model_uri}")
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        model = mlflow.sklearn.load_model(model_uri)  # type: ignore
        y_prob = model.predict_proba(X)[:, 1]  # type: ignore
    else:
        # For demo: use WoE test data with simple scoring
        logger.info("No model URI provided, using WoE test data for demo")
        woe_key = MINIO["paths"]["processed"] + "test_woe.csv"
        df_woe = load_csv(woe_key)
        y_true = df_woe[target_col].values
        X_woe = df_woe.drop(columns=[target_col]).fillna(0.0)

        # Simple scoring for demo (train a quick model)
        from sklearn.linear_model import LogisticRegression

        train_woe = load_csv(MINIO["paths"]["processed"] + "train_woe.csv")
        X_train = train_woe.drop(columns=[target_col]).fillna(0.0)
        y_train = train_woe[target_col]

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)

        # Align columns
        X_woe = X_woe.reindex(columns=X_train.columns, fill_value=0.0)
        y_prob = model.predict_proba(X_woe)[:, 1]

    # Calculate metrics
    logger.info("Calculating performance metrics...")
    metrics = calculate_all_metrics(y_true, y_prob, threshold)  # type: ignore
    status = evaluate_performance_status(metrics)

    # Calculate lift table
    lift_table = calculate_lift(y_true, y_prob)  # type: ignore

    # Add timestamp
    timestamp = datetime.now().isoformat()
    metrics["timestamp"] = timestamp

    # Save results
    results_df = pd.DataFrame([{**metrics, **{f"status_{k}": v for k, v in status.items()}}])
    upload_csv(results_df, final_output_key)

    lift_key = MINIO["paths"]["monitoring"] + "lift_table.csv"
    upload_csv(lift_table, lift_key)

    # Print report
    print("\n" + "-" * 35)
    print("MODEL PERFORMANCE MONITORING REPORT")
    print("-" * 35)
    print(f"Timestamp: {timestamp}")
    print(f"Data: {data_key} ({metrics['n_samples']} samples)")
    print(f"Default Rate: {metrics['default_rate']:.2%}")
    print("-" * 35)
    print("KEY METRICS:")
    print(f"  AUC:    {metrics['auc']:.4f}  [{status['auc']}]")
    print(f"  KS:     {metrics['ks']:.4f}  [{status['ks']}]")
    print(f"  Gini:   {metrics['gini']:.4f}  [{status['gini']}]")
    print("-" * 35)
    print("CLASSIFICATION METRICS (threshold={:.2f}):".format(threshold))
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print("-" * 35)
    print("CONFUSION MATRIX:")
    print(f"  TN: {metrics['tn']:,}  FP: {metrics['fp']:,}")
    print(f"  FN: {metrics['fn']:,}  TP: {metrics['tp']:,}")
    print(f"  Expected Cost: {metrics['expected_cost']:,.0f}")
    print("-" * 35)
    print(f"OVERALL STATUS: {status['overall']}")

    if status["overall"] == "CRITICAL":
        print("\n  ACTION REQUIRED: Model performance below minimum thresholds!")
    elif status["overall"] == "WARNING":
        print("\n  ATTENTION: Model performance approaching minimum thresholds")
    else:
        print("\n Model performance within acceptable range")

    print("-" * 35)

    return {"metrics": metrics, "status": status, "lift_table": lift_table}


if __name__ == "__main__":
    run_performance_monitoring()
