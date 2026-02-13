"""
Population Stability Index (PSI) Monitoring
PSI measures how much a population has shifted between two time periods.
PSI Interpretation:
- PSI < 0.1: No significant change
- 0.1 <= PSI < 0.25: Moderate change, investigate
- PSI >= 0.25: Significant change, model may need retraining
"""

import logging
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.io import get_s3_client, load_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load config

cfg = load_config()
MINIO = cfg["minio"]
BUCKET = MINIO["bucket"]

s3 = get_s3_client(cfg)

# PSI thresholds
PSI_THRESHOLD_WARNING = 0.1
PSI_THRESHOLD_CRITICAL = 0.25


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
    bucket_type: str = "quantile",
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate Population Stability Index between two distributions.
    Args:
        expected: Baseline distribution ( training data)
        actual: Current distribution ( production data)
        bins: Number of bins for discretization
        bucket_type: "quantile" or "fixed" binning strategy

    Returns:
        Tuple of (PSI value, detailed breakdown DataFrame)
    """
    # Remove nulls
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        logger.warning("Empty series passed to PSI calculation")
        return 0.0, pd.DataFrame()

    # Create bins based on expected (baseline) distribution
    if bucket_type == "quantile":
        try:
            _, bin_edges = pd.qcut(expected, q=bins, retbins=True, duplicates="drop")
        except ValueError:
            # Fall back to fewer bins if not enough unique values
            _, bin_edges = pd.qcut(
                expected,
                q=min(bins, expected.nunique()),
                retbins=True,
                duplicates="drop",
            )
    else:
        _, bin_edges = pd.cut(expected, bins=bins, retbins=True)

    # Ensure edge bins capture all values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Bin both distributions
    expected_binned = pd.cut(expected, bins=bin_edges, include_lowest=True)  # type: ignore
    actual_binned = pd.cut(actual, bins=bin_edges, include_lowest=True)  # type: ignore

    # Calculate proportions
    expected_counts = expected_binned.value_counts(sort=False, normalize=True)
    actual_counts = actual_binned.value_counts(sort=False, normalize=True)

    # Align indices
    all_bins = expected_counts.index.union(actual_counts.index)
    expected_pct = expected_counts.reindex(all_bins, fill_value=0.0)
    actual_pct = actual_counts.reindex(all_bins, fill_value=0.0)

    # Avoid division by zero with small epsilon
    eps = 1e-6
    expected_pct = expected_pct.replace(0, eps)
    actual_pct = actual_pct.replace(0, eps)

    # PSI formula: sum((actual% - expected%) * ln(actual% / expected%))
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = psi_values.sum()

    # Create detailed breakdown
    breakdown = pd.DataFrame(
        {
            "bin": [str(b) for b in all_bins],
            "expected_pct": expected_pct.values,
            "actual_pct": actual_pct.values,
            "psi_contribution": psi_values.values,
        }
    )

    return float(psi_total), breakdown


def calculate_feature_psi(
    df_expected: pd.DataFrame,
    df_actual: pd.DataFrame,
    features: Optional[List[str]] = None,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Calculate PSI for multiple features.
    Args:
        df_expected: Baseline DataFrame
        df_actual: Current DataFrame
        features: List of features to analyze (if None, use all numeric)
        bins: Number of bins

    Returns:
        DataFrame with PSI for each feature
    """
    if features is None:
        features = [
            c
            for c in df_expected.columns
            if df_expected[c].dtype != "object" and c in df_actual.columns
        ]

    results = []

    for feature in features:
        if feature not in df_expected.columns or feature not in df_actual.columns:
            continue

        try:
            psi, _ = calculate_psi(df_expected[feature], df_actual[feature], bins=bins)

            status = "OK"
            if psi >= PSI_THRESHOLD_CRITICAL:
                status = "CRITICAL"
            elif psi >= PSI_THRESHOLD_WARNING:
                status = "WARNING"

            results.append(
                {
                    "feature": feature,
                    "psi": round(psi, 4),
                    "status": status,
                    "expected_mean": round(df_expected[feature].mean(), 4),
                    "actual_mean": round(df_actual[feature].mean(), 4),
                    "expected_std": round(df_expected[feature].std(), 4),
                    "actual_std": round(df_actual[feature].std(), 4),
                }
            )
        except Exception as e:
            logger.warning(f"Could not calculate PSI for {feature}: {e}")
            continue

    return pd.DataFrame(results).sort_values("psi", ascending=False)


def calculate_score_psi(
    scores_expected: pd.Series, scores_actual: pd.Series, bins: int = 10
) -> Dict:
    """
    Calculate PSI specifically for model scores (PD scores).
    shifts significantly, the model's discrimination power may degrade.
    """
    psi, breakdown = calculate_psi(scores_expected, scores_actual, bins=bins)

    status = "OK"
    if psi >= PSI_THRESHOLD_CRITICAL:
        status = "CRITICAL - Model retraining recommended"
    elif psi >= PSI_THRESHOLD_WARNING:
        status = "WARNING - Investigate population shift"

    return {
        "psi": psi,
        "status": status,
        "expected_mean": float(scores_expected.mean()),
        "actual_mean": float(scores_actual.mean()),
        "expected_median": float(scores_expected.median()),
        "actual_median": float(scores_actual.median()),
        "breakdown": breakdown,
    }


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
    logger.info(f"Uploaded: {key}")


def run_psi_monitoring(
    baseline_key: str = None,  # type: ignore
    current_key: str = None,  # type: ignore
    output_key: str = None,  # type: ignore
) -> pd.DataFrame:
    """
    Run PSI monitoring comparing baseline to current data.

    Args:
        baseline_key: MinIO key for baseline data (default: training data)
        current_key: MinIO key for current data (default: test data for demo)
        output_key: MinIO key for output report

    Returns:
        DataFrame with PSI results
    """
    baseline_key = baseline_key or (MINIO["paths"]["processed"] + "train.csv")
    current_key = current_key or (MINIO["paths"]["processed"] + "test.csv")
    output_key = output_key or (MINIO["paths"]["monitoring"] + "mon1_psi_report.csv")

    logger.info(f"Loading baseline data: {baseline_key}")
    df_baseline = load_csv(baseline_key)

    logger.info(f"Loading current data: {current_key}")
    df_current = load_csv(current_key)

    logger.info("Calculating feature PSI...")
    psi_results = calculate_feature_psi(df_baseline, df_current)

    # Add timestamp
    psi_results["timestamp"] = datetime.now().isoformat()

    # Upload results
    upload_csv(psi_results, output_key)

    # Print summary
    print("\n" + "-" * 35)
    print("PSI MONITORING REPORT")
    print("-" * 35)
    print(f"Baseline: {baseline_key} ({len(df_baseline)} records)")
    print(f"Current: {current_key} ({len(df_current)} records)")
    print("-" * 35)

    critical_count = (psi_results["status"] == "CRITICAL").sum()
    warning_count = (psi_results["status"] == "WARNING").sum()

    if critical_count > 0:
        print(f"\n  CRITICAL: {critical_count} features with PSI >= {PSI_THRESHOLD_CRITICAL}")
        print(
            psi_results[psi_results["status"] == "CRITICAL"][["feature", "psi"]].to_string(
                index=False
            )
        )

    if warning_count > 0:
        print(f"\n WARNING: {warning_count} features with PSI >= {PSI_THRESHOLD_WARNING}")
        print(
            psi_results[psi_results["status"] == "WARNING"][["feature", "psi"]].to_string(
                index=False
            )
        )

    if critical_count == 0 and warning_count == 0:
        print("\n All features within acceptable PSI thresholds")

    print("-" * 35)
    print(f"Results saved to: {output_key}")
    print("-" * 35)

    return psi_results


if __name__ == "__main__":
    run_psi_monitoring()
