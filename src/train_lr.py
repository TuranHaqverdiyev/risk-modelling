"""
Logistic Regression Training Module for Credit Risk Model

This module trains and compares 4 logistic regression models:
1. Baseline (OHE) - OneHotEncoding for categoricals, StandardScaler for numerics
2. Simple WoE - Weight of Evidence transformation on raw features only
3. Advanced Features - Advanced engineered features without WoE
4. Advanced + WoE - Advanced engineered features with WoE transformation

All models are logged to MLflow with comprehensive metrics.
"""

import logging

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.transformers import SimpleWoETransformer
from src.utils.io import (
    get_s3_client,
    load_config,
    load_csv as _load_csv,
    upload_csv as _upload_csv,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET = "default_90p_12m"


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================


def ks_stat(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic.
    KS measures the maximum separation between cumulative distributions
    of good and bad customers. Higher is better (typically 0.2-0.5 for credit models).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Gini coefficient (2*AUC - 1)."""
    auc = roc_auc_score(y_true, y_prob)
    return float(2 * auc - 1)


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_fn: float, cost_fp: float) -> float:
    """
    Calculate expected cost of predictions.
    In credit risk:
    - FN (False Negative): Approved defaulter -> high cost
    - FP (False Positive): Rejected good customer -> opportunity cost
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(cost_fn * fn + cost_fp * fp)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> dict:
    """Calculate comprehensive metrics for model evaluation."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Core metrics
    auc = float(roc_auc_score(y_true, y_prob))
    ks = ks_stat(y_true, y_prob)
    gini = gini_coefficient(y_true, y_prob)

    # Classification metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))  # Sensitivity / TPR
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Derived metrics
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # TNR
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0  # False Negative Rate

    # Business metrics
    cost = expected_cost(y_true, y_pred, cost_fn, cost_fp)
    approval_rate = float((tn + fn) / len(y_true))  # Predicted negatives
    default_rate_approved = float(fn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return {
        # Discrimination
        "auc": auc,
        "ks": ks,
        "gini": gini,
        # Classification
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,  # Sensitivity / TPR
        "f1": f1,
        "specificity": specificity,  # TNR
        "fpr": fpr,
        "fnr": fnr,
        # Confusion matrix
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        # Business
        "expected_cost": cost,
        "approval_rate": approval_rate,
        "default_rate_approved": default_rate_approved,
    }


# =============================================================================
# I/O FUNCTIONS  (delegated to utils.io where possible)
# =============================================================================


def load_minio_csv(s3, bucket: str, key: str) -> pd.DataFrame:  # noqa: ARG001
    """Load CSV from MinIO (thin wrapper kept for call-site compat)."""
    return _load_csv(key, cfg)


def upload_text(s3, bucket: str, key: str, text: str) -> None:  # noqa: ARG001
    """Upload text to MinIO."""
    _s3 = get_s3_client(cfg)
    _s3.put_object(Bucket=cfg["minio"]["bucket"], Key=key, Body=text.encode("utf-8"))


def upload_csv(s3, bucket: str, key: str, df: pd.DataFrame) -> None:  # noqa: ARG001
    """Upload DataFrame as CSV to MinIO."""
    _upload_csv(df, key, cfg)


# =============================================================================
# CONFIGURATION
# =============================================================================

cfg = load_config()

with open("configs/costs.yaml") as f:
    costs_cfg = yaml.safe_load(f)

MINIO = cfg["minio"]
BUCKET = MINIO["bucket"]

s3 = get_s3_client(cfg)

MLFLOW_URI = cfg["mlflow"]["tracking_uri"]
mlflow.set_tracking_uri(MLFLOW_URI)

# Input paths - Raw data
TRAIN_KEY_RAW = MINIO["paths"]["processed"] + "train.csv"
TEST_KEY_RAW = MINIO["paths"]["processed"] + "test.csv"

# Input paths - Advanced features (all features, no selection, no WoE)
TRAIN_KEY_ENHANCED = MINIO["paths"]["processed"] + "train_enhanced.csv"
TEST_KEY_ENHANCED = MINIO["paths"]["processed"] + "test_enhanced.csv"

# Input paths - Selected features (IV + Corr + VIF filtered, no WoE)
TRAIN_KEY_SELECTED = MINIO["paths"]["processed"] + "train_selected.csv"
TEST_KEY_SELECTED = MINIO["paths"]["processed"] + "test_selected.csv"

# Input paths - WoE transformed (selected features + WoE)
TRAIN_KEY_WOE = MINIO["paths"]["processed"] + "train_woe.csv"
TEST_KEY_WOE = MINIO["paths"]["processed"] + "test_woe.csv"

# Output paths
REPORT_KEY = MINIO["paths"]["outputs"] + "ds3_lr_report.txt"
SUMMARY_KEY = MINIO["paths"]["outputs"] + "ds3_lr_summary.csv"
DETAILED_KEY = MINIO["paths"]["outputs"] + "ds3_lr_detailed_metrics.csv"

# Costs from centralized config
COST_FN = costs_cfg["costs"]["false_negative"]
COST_FP = costs_cfg["costs"]["false_positive"]
DEFAULT_THRESHOLD = costs_cfg["threshold"]["default"]


# =============================================================================
# SIMPLE WOE TRANSFORMER (for raw features only)
# =============================================================================


# =============================================================================
# TRAINING FUNCTION
# =============================================================================


def train_and_log_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_name: str,
    model_type: str,
    preprocessor=None,
    lr_cv: bool = False,
    penalty: str = "l2",
    solver: str = "lbfgs",
    Cs: list = None,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Train a logistic regression model and log to MLflow.

    Returns dictionary with all metrics and run info.
    """

    with mlflow.start_run(run_name=run_name) as run:
        # Cross-validated Logistic Regression
        if Cs is None:
            Cs = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        lr_cv_model = (
            LogisticRegressionCV(
                penalty=penalty,
                solver=solver,
                Cs=Cs,
                cv=cv,
                scoring="roc_auc",
                class_weight="balanced",
                max_iter=1000,
                n_jobs=1 if solver == "liblinear" else -1,
                random_state=random_state,
                refit=True,
            )
            if lr_cv
            else LogisticRegression(
                penalty=penalty,
                solver=solver,
                max_iter=1000,
                n_jobs=1 if solver == "liblinear" else -1,
                class_weight="balanced",
                random_state=random_state,
            )
        )

        if preprocessor is not None:
            model = Pipeline([("preprocessor", preprocessor), ("classifier", lr_cv_model)])
        else:
            model = lr_cv_model

        # Train
        logger.info(f"Training {run_name} (CV={lr_cv})...")
        model.fit(X_train, y_train)

        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= DEFAULT_THRESHOLD).astype(int)

        # Convert to numpy
        y_test_arr: np.ndarray = np.asarray(y_test)

        # Calculate all metrics
        metrics = calculate_all_metrics(y_test_arr, y_prob, y_pred, COST_FN, COST_FP)

        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("threshold", DEFAULT_THRESHOLD)
        mlflow.log_param("cost_fn", COST_FN)
        mlflow.log_param("cost_fp", COST_FP)
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("n_test", len(y_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("solver", solver)
        if lr_cv:
            if hasattr(model, "C_"):
                mlflow.log_param("C", float(model.C_[0]))
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        # Log model
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
            logger.info("Model artifact logged to MLflow")
        except Exception as e:
            logger.warning(f"Could not log model to MLflow: {e}")

        # Log feature coefficients
        try:
            if isinstance(model, Pipeline):
                classifier = model.named_steps["classifier"]
            else:
                classifier = model

            if hasattr(classifier, "coef_"):
                coef_df = pd.DataFrame(
                    {
                        "feature": X_train.columns,
                        "coefficient": classifier.coef_[0],
                        "abs_coefficient": np.abs(classifier.coef_[0]),
                    }
                ).sort_values("abs_coefficient", ascending=False)
                mlflow.log_text(coef_df.to_csv(index=False), "feature_coefficients.csv")
        except Exception as e:
            logger.warning(f"Could not log coefficients: {e}")

        # Add run info
        metrics["run_id"] = run.info.run_id
        metrics["run_name"] = run_name
        metrics["model_type"] = model_type
        metrics["n_features"] = X_train.shape[1]

        # Return raw predictions alongside metrics (F12)
        metrics["_y_true"] = y_test_arr
        metrics["_y_prob"] = y_prob
        metrics["_y_pred"] = y_pred

        # Also return train predictions for ROC comparison (F12)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        metrics["_y_true_train"] = np.asarray(y_train)
        metrics["_y_prob_train"] = y_prob_train
        metrics["_y_pred_train"] = (y_prob_train >= DEFAULT_THRESHOLD).astype(int)

        logger.info(
            f"{run_name}: AUC={metrics['auc']:.4f}, KS={metrics['ks']:.4f}, "
            f"F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}"
        )

        return metrics


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================


def main():
    """Main training pipeline with 4 models."""

    logger.info("=" * 60)
    logger.info("CREDIT RISK MODEL TRAINING - 4 MODEL COMPARISON")
    logger.info("=" * 60)

    # Setup MLflow experiment
    exp_name = f"{cfg['mlflow']['experiment_base']}"
    mlflow.set_experiment(exp_name)
    logger.info(f"MLflow experiment: {exp_name}")

    all_metrics = []

    # =========================================================================
    # Load all datasets
    # =========================================================================
    logger.info("\nLoading datasets...")

    # Raw data
    train_raw = load_minio_csv(s3, BUCKET, TRAIN_KEY_RAW)
    test_raw = load_minio_csv(s3, BUCKET, TEST_KEY_RAW)
    logger.info(f"Raw data: train={train_raw.shape}, test={test_raw.shape}")

    # Selected features data (IV + Corr + VIF filtered, no WoE) - for Model 3
    try:
        train_selected = load_minio_csv(s3, BUCKET, TRAIN_KEY_SELECTED)
        test_selected = load_minio_csv(s3, BUCKET, TEST_KEY_SELECTED)
        logger.info(
            f"Selected features data: train={train_selected.shape}, test={test_selected.shape}"
        )
        has_selected = True
    except Exception as e:
        logger.warning(f"Selected features data not found: {e}")
        has_selected = False

    # WoE data (selected features + WoE transformation) - for Model 4
    try:
        train_woe = load_minio_csv(s3, BUCKET, TRAIN_KEY_WOE)
        test_woe = load_minio_csv(s3, BUCKET, TEST_KEY_WOE)
        logger.info(f"WoE data: train={train_woe.shape}, test={test_woe.shape}")
        has_woe = True
    except Exception as e:
        logger.warning(f"WoE data not found: {e}")
        has_woe = False

    # =========================================================================
    # MODEL 1: Baseline (OHE + StandardScaler)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 1: Baseline (OHE + StandardScaler)")
    logger.info("=" * 60)

    X_tr_raw = train_raw.drop(columns=[TARGET])
    y_tr_raw = train_raw[TARGET]
    X_te_raw = test_raw.drop(columns=[TARGET])
    y_te_raw = test_raw[TARGET]

    cat_cols = [c for c in X_tr_raw.columns if X_tr_raw[c].dtype == "object"]
    num_cols = [c for c in X_tr_raw.columns if X_tr_raw[c].dtype != "object"]

    preprocessor_baseline = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    metrics_baseline = train_and_log_model(
        X_tr_raw,
        y_tr_raw,
        X_te_raw,
        y_te_raw,
        run_name="1_Baseline_OHE",
        model_type="baseline",
        preprocessor=preprocessor_baseline,
        lr_cv=True,
        penalty="l2",
        solver="lbfgs",
    )
    all_metrics.append(metrics_baseline)

    # =========================================================================
    # MODEL 2: Simple WoE (raw features only)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 2: Simple WoE (raw features only)")
    logger.info("=" * 60)

    # Apply simple WoE to raw numeric features
    woe_transformer = SimpleWoETransformer(bins=5)
    woe_transformer.fit(X_tr_raw, y_tr_raw)

    X_tr_simple_woe = woe_transformer.transform(X_tr_raw)
    X_te_simple_woe = woe_transformer.transform(X_te_raw)

    # Align columns
    X_te_simple_woe = X_te_simple_woe.reindex(columns=X_tr_simple_woe.columns, fill_value=0.0)

    metrics_simple_woe = train_and_log_model(
        X_tr_simple_woe.fillna(0),
        y_tr_raw,
        X_te_simple_woe.fillna(0),
        y_te_raw,
        run_name="2_Simple_WoE",
        model_type="simple_woe",
        preprocessor=None,
        lr_cv=True,
        penalty="l2",
        solver="lbfgs",
    )
    all_metrics.append(metrics_simple_woe)

    # =========================================================================
    # MODEL 3: Advanced Features with Best Practice Selection (no WoE)
    # Features: IV filtered + Correlation removed + VIF removed
    # =========================================================================
    if has_selected:
        logger.info("\n" + "=" * 60)
        logger.info("MODEL 3: Advanced Features (IV + Corr + VIF Selected, no WoE)")
        logger.info("=" * 60)

        X_tr_selected = train_selected.drop(columns=[TARGET])
        y_tr_selected = train_selected[TARGET]
        X_te_selected = test_selected.drop(columns=[TARGET])
        y_te_selected = test_selected[TARGET]

        # Only numeric features, with scaling
        num_cols_selected = X_tr_selected.select_dtypes(include=[np.number]).columns.tolist()

        preprocessor_selected = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        metrics_advanced = train_and_log_model(
            X_tr_selected[num_cols_selected],
            y_tr_selected,
            X_te_selected[num_cols_selected],
            y_te_selected,
            run_name="3_Advanced_Selected",
            model_type="advanced_selected",
            preprocessor=preprocessor_selected,
            lr_cv=True,
            penalty="l2",
            solver="lbfgs",
        )
        all_metrics.append(metrics_advanced)
    else:
        logger.warning("Skipping Model 3 - selected features data not available")

    # =========================================================================
    # MODEL 4: Advanced Features + WoE (Best Practice Pipeline)
    # Features: Same selected features, but WoE transformed
    # =========================================================================
    if has_woe:
        logger.info("\n" + "=" * 60)
        logger.info("MODEL 4: Advanced Features + WoE")
        logger.info("=" * 60)

        X_tr_woe = train_woe.drop(columns=[TARGET]).fillna(0.0)
        y_tr_woe = train_woe[TARGET]
        X_te_woe = test_woe.drop(columns=[TARGET]).fillna(0.0)
        y_te_woe = test_woe[TARGET]

        # Align columns
        X_te_woe = X_te_woe.reindex(columns=X_tr_woe.columns, fill_value=0.0)

        metrics_advanced_woe = train_and_log_model(
            X_tr_woe,
            y_tr_woe,
            X_te_woe,
            y_te_woe,
            run_name="4_Advanced_WoE",
            model_type="advanced_woe",
            preprocessor=None,
            lr_cv=True,
            penalty="l2",
            solver="lbfgs",
        )
        all_metrics.append(metrics_advanced_woe)
    else:
        logger.warning("Skipping Model 4 - WoE data not available")

    # =========================================================================
    # PERSIST PER-MODEL PREDICTIONS (MRM-F12)
    # =========================================================================
    logger.info("\nPersisting per-model predictions to MinIO...")
    for m in all_metrics:
        # Save Test Predictions (Default)
        pred_df_test = pd.DataFrame(
            {
                "y_true": m.get("_y_true"),
                "y_prob": m.get("_y_prob"),
                "y_pred": m.get("_y_pred"),
            }
        )
        pred_key_test = MINIO["paths"]["outputs"] + f"ds3_lr_predictions_{m['run_name']}.csv"
        _upload_csv(pred_df_test, pred_key_test, cfg)

        # Save Train Predictions (for ROC comparison)
        if "_y_true_train" in m:
            pred_df_train = pd.DataFrame(
                {
                    "y_true": m.get("_y_true_train"),
                    "y_prob": m.get("_y_prob_train"),
                    "y_pred": m.get("_y_pred_train"),
                }
            )
            pred_key_train = (
                MINIO["paths"]["outputs"] + f"ds3_lr_predictions_{m['run_name']}_train.csv"
            )
            _upload_csv(pred_df_train, pred_key_train, cfg)
            logger.info(f"  Saved predictions (Train/Test) for: {m['run_name']}")
        else:
            logger.info(f"  Saved predictions (Test only) for: {m['run_name']}")

    # =========================================================================
    # PRECOMPUTE LEARNING CURVE (MRM-F8)
    # =========================================================================
    logger.info("\nPrecomputing learning curve...")
    try:
        from sklearn.model_selection import learning_curve as sk_learning_curve

        # Use the best WoE model data for the learning curve
        if has_woe:
            X_lc = train_woe.drop(columns=[TARGET]).fillna(0.0)
            y_lc = train_woe[TARGET]
        else:
            X_lc = X_tr_raw.select_dtypes(include=[np.number]).fillna(0)
            y_lc = y_tr_raw

        lr_lc = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )
        train_sizes, train_scores, val_scores = sk_learning_curve(
            lr_lc,
            X_lc,
            y_lc,
            cv=5,
            scoring="roc_auc",
            train_sizes=np.linspace(0.1, 1.0, 8),
            shuffle=True,
            random_state=42,
        )[:3]

        lc_df = pd.DataFrame(
            {
                "train_size": train_sizes,
                "train_auc_mean": np.mean(train_scores, axis=1),
                "train_auc_std": np.std(train_scores, axis=1),
                "val_auc_mean": np.mean(val_scores, axis=1),
                "val_auc_std": np.std(val_scores, axis=1),
            }
        )
        lc_key = MINIO["paths"]["outputs"] + "ds3_learning_curve.csv"
        _upload_csv(lc_df, lc_key, cfg)
        logger.info(f"Learning curve saved: {lc_key}")
    except Exception as e:
        logger.warning(f"Could not precompute learning curve: {e}")

    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING REPORTS")
    logger.info("=" * 60)

    report_text = f"""
================================================================================
LOGISTIC REGRESSION MODEL COMPARISON REPORT
================================================================================

Experiment: {exp_name}
Training samples: {len(y_tr_raw)}
Test samples: {len(y_te_raw)}
Cost parameters: FN={COST_FN}, FP={COST_FP}
Default threshold: {DEFAULT_THRESHOLD}

================================================================================
MODEL SUMMARY
================================================================================

| Model                  | AUC    | KS     | Gini   | F1     | Precision | Recall |
|------------------------|--------|--------|--------|--------|-----------|--------|
"""

    for m in all_metrics:
        report_text += (
            f"| {m['run_name']:<22} | {m['auc']:.4f} | {m['ks']:.4f} | "
            f"{m['gini']:.4f} | {m['f1']:.4f} | {m['precision']:.4f}    | "
            f"{m['recall']:.4f} |\n"
        )

    report_text += """
================================================================================
DETAILED METRICS PER MODEL
================================================================================
"""

    for m in all_metrics:
        report_text += f"""
--------------------------------------------------------------------------------
{m["run_name"]} ({m["n_features"]} features)
--------------------------------------------------------------------------------
DISCRIMINATION METRICS:
  - ROC AUC:        {m["auc"]:.4f}
  - KS Statistic:   {m["ks"]:.4f}
  - Gini:           {m["gini"]:.4f}

CLASSIFICATION METRICS:
  - Accuracy:       {m["accuracy"]:.4f}
  - Precision:      {m["precision"]:.4f}  (TP / (TP + FP))
  - Recall (TPR):   {m["recall"]:.4f}  (TP / (TP + FN)) - Sensitivity
  - Specificity:    {m["specificity"]:.4f}  (TN / (TN + FP)) - TNR
  - F1 Score:       {m["f1"]:.4f}
  - FPR:            {m["fpr"]:.4f}  (FP / (FP + TN))
  - FNR:            {m["fnr"]:.4f}  (FN / (FN + TP))

CONFUSION MATRIX:
                    Predicted
                    Neg     Pos
  Actual Neg       {m["tn"]:>6}  {m["fp"]:>6}
  Actual Pos       {m["fn"]:>6}  {m["tp"]:>6}

BUSINESS METRICS:
  - Expected Cost:         {m["expected_cost"]:.2f}
  - Approval Rate:         {m["approval_rate"]:.2%}
  - Default Rate Approved: {m["default_rate_approved"]:.2%}

MLflow Run ID: {m["run_id"]}
"""

    report_text += """
================================================================================
MODELS INTERPRETATION
================================================================================
1. Baseline (OHE): Basic model with one-hot encoding - establishes minimum benchmark
2. Simple WoE: WoE on raw features only - tests WoE benefit without feature engineering
3. Advanced Features: 83 engineered features (no WoE) - tests feature engineering benefit
4. Advanced + WoE: Full pipeline - combines feature engineering with WoE transformation

RECOMMENDATIONS:
- Compare AUC/KS to see discrimination improvement
- Check Recall (TPR) for defaulter detection capability
- Review Expected Cost for business impact
- Higher approval rate with lower default rate = better model
"""

    # Save report
    upload_text(s3, BUCKET, REPORT_KEY, report_text)
    print(report_text)
    logger.info(f"Report saved to: {REPORT_KEY}")

    # Save summary CSV
    summary_data = []
    for m in all_metrics:
        summary_data.append(
            {
                "model": m["run_name"],
                "model_type": m["model_type"],
                "n_features": m["n_features"],
                "auc": m["auc"],
                "ks": m["ks"],
                "gini": m["gini"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "fpr": m["fpr"],
                "fnr": m["fnr"],
                "expected_cost": m["expected_cost"],
                "approval_rate": m["approval_rate"],
                "default_rate_approved": m["default_rate_approved"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
                "run_id": m["run_id"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    upload_csv(s3, BUCKET, SUMMARY_KEY, summary_df)
    upload_csv(s3, BUCKET, DETAILED_KEY, summary_df)
    logger.info(f"Summary CSV saved to: {SUMMARY_KEY}")
    logger.info(f"Detailed CSV saved to: {DETAILED_KEY}")

    # Print comparison table to console
    print("\n" + "=" * 80)
    print("QUICK COMPARISON")
    print("=" * 80)
    print(
        summary_df[["model", "auc", "ks", "f1", "recall", "precision", "expected_cost"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
