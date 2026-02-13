"""
Model Interpretation Module
Provides interpretability tools for credit risk models:
- Feature contribution analysis
- Partial dependence plots data
- Individual prediction explanations
- Global feature effects

For logistic regression with WoE features, interpretation is straightforward:
- Coefficient * WoE value = contribution to log-odds
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

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

# MinIO client
s3 = get_s3_client(cfg)


class ModelInterpreter:
    """
    Interpreter for WoE-based logistic regression models.
    For WoE models, interpretation is relatively simple:
    - Each feature contributes: coefficient * WoE_value to log-odds
    - Positive contribution → increases P(default)
    - Negative contribution → decreases P(default)
    """

    def __init__(
        self,
        coefficients: pd.DataFrame,
        woe_bins: Optional[Dict] = None,
        intercept: float = 0.0,
    ):
        """
        Initialize interpreter.
        Args:
            coefficients: DataFrame with 'feature' and 'coefficient' columns
            woe_bins: Dictionary with WoE bin mappings (from WoETransformer)
            intercept: Model intercept term
        """
        self.coef_dict = dict(
            zip(coefficients["feature"], coefficients["coefficient"], strict=True)
        )
        self.woe_bins = woe_bins or {}
        self.intercept = intercept

    def explain_prediction(self, row: pd.Series, top_n: int = 10) -> Tuple[pd.DataFrame, Dict]:
        """
        Explain a single prediction by breaking down feature contributions
        Args:
            row: Single observation (WoE-encoded)
            top_n: Number of top features to return

        Returns:
            DataFrame with feature contributions
        """
        contributions = []

        for feature, value in row.items():
            if feature in self.coef_dict and feature != TARGET:
                coef = self.coef_dict[feature]
                contribution = coef * value
                contributions.append(
                    {
                        "feature": feature,
                        "woe_value": value,
                        "coefficient": coef,
                        "contribution": contribution,
                        "direction": ("Risk Higher" if contribution > 0 else "Risk Lower"),
                    }
                )

        df = pd.DataFrame(contributions)
        df = df.sort_values("contribution", key=abs, ascending=False)

        # Add total and intercept
        total_contribution = df["contribution"].sum()
        log_odds = self.intercept + total_contribution
        probability = 1 / (1 + np.exp(-log_odds))

        df["cumulative"] = df["contribution"].cumsum()

        return df.head(top_n), {
            "intercept": self.intercept,
            "total_contribution": total_contribution,
            "log_odds": log_odds,
            "probability": probability,
        }

    def feature_effects(self, X: pd.DataFrame, feature: str, n_points: int = 20) -> pd.DataFrame:
        """
        Calculate average effect of a feature across its range.
        This is similar to a Partial Dependence Plot.
        Args:
            X: WoE-encoded feature matrix
            feature: Feature name to analyze
            n_points: Number of points to evaluate

        Returns:
            DataFrame with feature values and average effects
        """
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found")

        if feature not in self.coef_dict:
            raise ValueError(f"No coefficient for feature '{feature}'")

        coef = self.coef_dict[feature]

        # Get range of values in the data
        min_val = X[feature].min()
        max_val = X[feature].max()

        values = np.linspace(min_val, max_val, n_points)
        effects = values * coef

        return pd.DataFrame(
            {
                "woe_value": values,
                "contribution": effects,
                "direction": ["Risk Higher" if e > 0 else "Risk Lower" for e in effects],
            }
        )

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate global feature importance based on mean absolute contribution.
        Args:
            X: WoE-encoded feature matrix

        Returns:
            DataFrame with global importance scores
        """
        importance = []

        for feature in X.columns:
            if feature in self.coef_dict and feature != TARGET:
                coef = self.coef_dict[feature]
                mean_abs_contribution = (X[feature] * coef).abs().mean()
                std_contribution = (X[feature] * coef).std()

                importance.append(
                    {
                        "feature": feature,
                        "coefficient": coef,
                        "mean_woe": X[feature].mean(),
                        "mean_abs_contribution": mean_abs_contribution,
                        "std_contribution": std_contribution,
                        "risk_direction": ("Increases Risk" if coef > 0 else "Decreases Risk"),
                    }
                )

        df = pd.DataFrame(importance)
        return df.sort_values("mean_abs_contribution", ascending=False)


def generate_interpretation_report(
    test_woe_key: str, importance_key: str, output_key: str, sample_size: int = 100
) -> None:
    """
    Generate comprehensive interpretation report.
    Args:
        test_woe_key: Path to test WoE data
        importance_key: Path to feature importance CSV
        output_key: Path for output report
        sample_size: Number of samples to explain
    """
    logger.info("Loading data...")
    df_test = _load_csv(test_woe_key, cfg)
    df_importance = _load_csv(importance_key, cfg)

    X = df_test.drop(columns=[TARGET], errors="ignore")
    X = X.fillna(0.0)

    # Initialize interpreter
    interpreter = ModelInterpreter(df_importance)

    # Generate global importance
    logger.info("Calculating global importance...")
    global_imp = interpreter.global_importance(X)

    # Generate sample explanations
    logger.info(f"Generating explanations for {sample_size} samples...")
    sample_idx = np.random.choice(len(X), min(sample_size, len(X)), replace=False)

    all_explanations = []
    for idx in sample_idx:
        row = X.iloc[idx]
        exp_df, summary = interpreter.explain_prediction(row, top_n=5)
        exp_df["sample_idx"] = idx
        exp_df["predicted_prob"] = summary["probability"]
        all_explanations.append(exp_df)

    explanations_df = pd.concat(all_explanations, ignore_index=True)

    # Save results
    _upload_csv(global_imp, output_key.replace(".csv", "_global.csv"), cfg)
    _upload_csv(explanations_df, output_key.replace(".csv", "_samples.csv"), cfg)

    logger.info("Interpretation report saved")


def main():
    """Generate interpretation outputs."""
    paths = MINIO["paths"]

    test_woe_key = paths["processed"] + "test_woe.csv"
    importance_key = paths["outputs"] + "ds4_feature_importance.csv"
    output_key = paths["outputs"] + "ds6_interpretation.csv"

    generate_interpretation_report(
        test_woe_key=test_woe_key, importance_key=importance_key, output_key=output_key
    )

    print("-" * 35)
    print("MODEL INTERPRETATION")
    print("-" * 35)
    print(f"Global importance: {output_key.replace('.csv', '_global.csv')}")
    print(f"Sample explanations: {output_key.replace('.csv', '_samples.csv')}")
    print("-" * 35)


if __name__ == "__main__":
    main()
