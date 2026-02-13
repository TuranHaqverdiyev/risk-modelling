"""
Feature Importance Analysis Module
Extracts and analyzes feature importance from trained models:
- Logistic regression coefficients
- Standardized coefficients for comparability
- Risk direction interpretation
- Feature ranking and selection insights
"""

import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.io import (
    get_s3_client,
    load_config,
    load_csv as _load_csv,
    upload_csv as _upload_csv,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET = "default_90p_12m"

# Load config

cfg = load_config()
MINIO = cfg["minio"]
BUCKET = MINIO["bucket"]

TRAIN_WOE_KEY = MINIO["paths"]["processed"] + "train_woe.csv"
OUTPUT_KEY = MINIO["paths"]["outputs"] + "ds4_feature_importance.csv"

# MinIO client
s3 = get_s3_client(cfg)


def try_load_model_from_mlflow() -> tuple:
    """
    Try to load the trained WoE model from MLflow.
    Returns:
        Tuple of (model, success_flag)
    """
    try:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        model = mlflow.sklearn.load_model("models:/credit_risk_woe/latest")  # type: ignore
        logger.info("Loaded model from MLflow: credit_risk_woe")
        return model, True
    except Exception as e:
        logger.warning(f"Could not load from MLflow: {e}")
        return None, False


def extract_coefficients(model, feature_names: list) -> pd.DataFrame:
    """
    Extract coefficients from a logistic regression model.
    Args:
        model: Trained LogisticRegression model
        feature_names: List of feature names

    Returns:
        DataFrame with coefficient analysis
    """
    coef = model.coef_[0]

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
            "exp_coefficient": np.exp(coef),  # Odds ratio
        }
    )

    # Rank by absolute value
    importance = importance.sort_values("abs_coefficient", ascending=False)
    importance["rank"] = range(1, len(importance) + 1)

    # Risk direction interpretation
    # - Positive coefficient: Higher WoE -> Higher P(default) -> High risk
    # - Negative coefficient: Higher WoE -> Lower P(default) -> Low risk
    importance["risk_direction"] = importance["coefficient"].apply(
        lambda x: "Increases Risk" if x > 0 else "Decreases Risk"
    )

    # Relative importance (normalized to sum to 100%)
    importance["relative_importance"] = (
        importance["abs_coefficient"] / importance["abs_coefficient"].sum() * 100
    )

    # Cumulative importance
    importance["cumulative_importance"] = importance["relative_importance"].cumsum()

    return importance


def calculate_standardized_coefficients(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Calculate standardized coefficients for fair comparison.

    Standardized coefficients account for different feature scales.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_scaled, y)

    standardized = pd.DataFrame(
        {
            "feature": X.columns,
            "standardized_coef": model.coef_[0],
            "abs_standardized": np.abs(model.coef_[0]),
        }
    )

    return standardized.sort_values("abs_standardized", ascending=False)


def main():
    """Main feature importance analysis."""

    logger.info("Loading WoE training data...")
    df = _load_csv(TRAIN_WOE_KEY, cfg)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Handle missing values
    X = X.fillna(0.0)
    feature_names = X.columns.tolist()

    # Try to load model from MLflow first
    model, from_mlflow = try_load_model_from_mlflow()

    if not from_mlflow:
        # Train a model locally if MLflow is not available
        logger.info("Training local model for coefficient extraction...")
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X, y)

    # Extract coefficients
    logger.info("Extracting feature coefficients...")
    importance = extract_coefficients(model, feature_names)

    # Also calculate standardized coefficients
    logger.info("Calculating standardized coefficients...")
    standardized = calculate_standardized_coefficients(X, y)

    # Merge standardized coefficients
    importance = importance.merge(
        standardized[["feature", "standardized_coef", "abs_standardized"]],
        on="feature",
        how="left",
    )

    # Save results
    _upload_csv(importance, OUTPUT_KEY, cfg)

    # Print report
    print("\n" + "-" * 35)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("-" * 35)
    print(f"Model source: {'MLflow' if from_mlflow else 'Local training'}")
    print(f"Total features: {len(importance)}")
    print("-" * 35)

    print("\nTOP 10 FEATURES BY IMPORTANCE:")
    print("-" * 35)
    top10 = importance.head(10)
    for _, row in top10.iterrows():
        print(
            f"  {row['rank']:2d}. {row['feature']:<30} "
            f"Coef: {row['coefficient']:>8.4f} "
            f"({row['risk_direction']})"
        )

    print("-" * 35)
    print("\nFEATURE SELECTION INSIGHTS:")

    # Features covering 80% of importance
    n_features_80 = (importance["cumulative_importance"] <= 80).sum() + 1
    print(f"  - {n_features_80} features cover 80% of total importance")

    # Features with very low importance
    low_importance = (importance["relative_importance"] < 1).sum()
    print(f"  - {low_importance} features contribute < 1% each")

    # Risk-increasing vs decreasing features
    n_increasing = (importance["coefficient"] > 0).sum()
    n_decreasing = (importance["coefficient"] < 0).sum()
    print(f"  - {n_increasing} features increase risk")
    print(f"  - {n_decreasing} features decrease risk")

    print("-" * 35)
    print(f"Results saved to: {OUTPUT_KEY}")
    print("-" * 35)


if __name__ == "__main__":
    main()
