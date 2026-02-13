"""
Data Analytics Pipeline Orchestrator
This script runs the data analytics / EDA pipeline:
1. Dataset Overview & Profiling
2. Income & Employment Analysis
3. DPD Behavior Analysis
4. Credit Inquiry Analysis
5. Affordability Analysis
These analyses provide business insights before model development.

Usage:
    python pipelines/run_analytics.py [--report-only]
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd

from src.utils.io import get_s3_client, load_config


cfg = load_config()
MINIO = cfg["minio"]
BUCKET = MINIO["bucket"]

s3 = get_s3_client(cfg)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("analytics_pipeline")


# Config and S3 client are now initialized above from src.utils.io


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


def dataset_overview(df: pd.DataFrame, target_col: str = "default_90p_12m") -> dict:
    """
    Generate dataset overview statistics.
    """
    logger.info("Running dataset overview analysis...")

    overview = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_numeric": len(df.select_dtypes(include=[np.number]).columns),
        "n_categorical": len(df.select_dtypes(include=["object"]).columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "missing_total": df.isnull().sum().sum(),
        "missing_pct": (df.isnull().sum().sum() / df.size) * 100,
        "duplicate_rows": df.duplicated().sum(),
    }

    if target_col in df.columns:
        overview["default_rate"] = df[target_col].mean()
        overview["default_count"] = df[target_col].sum()
        overview["non_default_count"] = len(df) - df[target_col].sum()

    # Column-level missing analysis
    missing_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isnull().sum().values,
            "missing_pct": (df.isnull().sum() / len(df) * 100).values,
            "dtype": df.dtypes.astype(str).values,
        }
    ).sort_values("missing_pct", ascending=False)

    return {"overview": overview, "missing_analysis": missing_df}


def univariate_analysis(df: pd.DataFrame, target_col: str = "default_90p_12m") -> pd.DataFrame:
    """
    Perform univariate analysis for all features.
    """
    logger.info("Running univariate analysis...")

    results = []

    for col in df.columns:
        if col == target_col:
            continue

        stats = {
            "feature": col,
            "dtype": str(df[col].dtype),
            "n_unique": df[col].nunique(),
            "missing_pct": df[col].isnull().mean() * 100,
        }

        if df[col].dtype != "object":
            stats.update(
                {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "p25": df[col].quantile(0.25),
                    "median": df[col].median(),
                    "p75": df[col].quantile(0.75),
                    "max": df[col].max(),
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis(),
                }
            )

            # Default rate correlation
            if target_col in df.columns:
                valid_mask = df[col].notna()
                if valid_mask.sum() > 10:
                    stats["default_corr"] = df.loc[valid_mask, col].corr(
                        df.loc[valid_mask, target_col]
                    )
        else:
            stats["top_value"] = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
            stats["top_freq"] = (
                (df[col] == stats["top_value"]).mean() if stats["top_value"] else None
            )

        results.append(stats)

    return pd.DataFrame(results)


def bivariate_analysis(df: pd.DataFrame, target_col: str = "default_90p_12m") -> pd.DataFrame:
    """
    Analyze relationship between features and target.
    """
    logger.info("Running bivariate analysis...")

    if target_col not in df.columns:
        logger.warning(f"Target column {target_col} not found")
        return pd.DataFrame()

    results = []

    for col in df.columns:
        if col == target_col:
            continue

        stats = {"feature": col}

        if df[col].dtype == "object":
            # Categorical: default rate by category
            group_stats = df.groupby(col)[target_col].agg(["mean", "count"])
            stats["n_categories"] = len(group_stats)  # type: ignore
            stats["max_default_rate"] = group_stats["mean"].max()
            stats["min_default_rate"] = group_stats["mean"].min()
            stats["default_rate_range"] = stats["max_default_rate"] - stats["min_default_rate"]
        else:
            # Numeric: binned default rates
            try:
                df["_bin"] = pd.qcut(df[col], q=5, duplicates="drop")
                group_stats = df.groupby("_bin", observed=True)[target_col].agg(["mean", "count"])
                df.drop("_bin", axis=1, inplace=True)

                stats["n_bins"] = len(group_stats)  # type: ignore
                stats["max_default_rate"] = group_stats["mean"].max()
                stats["min_default_rate"] = group_stats["mean"].min()
                stats["default_rate_range"] = stats["max_default_rate"] - stats["min_default_rate"]
                stats["monotonic"] = (  # type: ignore
                    (group_stats["mean"].is_monotonic_increasing)
                    or (group_stats["mean"].is_monotonic_decreasing)
                )
            except Exception:
                pass

        results.append(stats)

    return pd.DataFrame(results)


def run_analytics_pipeline(report_only: bool = False) -> dict:
    """
    Run the complete analytics pipeline.
    """

    logger.info("\n" + "-" * 35)
    logger.info("ABB CREDIT RISK - DATA ANALYTICS PIPELINE")
    logger.info("-" * 35)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("-" * 35 + "\n")

    pipeline_start = datetime.now()
    results = {"started_at": pipeline_start.isoformat(), "analyses": {}}

    try:
        # Load raw data
        raw_key = MINIO["paths"]["raw"] + "dataset.xlsx"
        logger.info(f"Loading raw data from: {raw_key}")

        obj = s3.get_object(Bucket=BUCKET, Key=raw_key)
        df = pd.read_excel(BytesIO(obj["Body"].read()))

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        # Analysis 1: Dataset Overview
        overview = dataset_overview(df)
        results["analyses"]["overview"] = overview["overview"]

        # Save missing analysis
        upload_csv(
            overview["missing_analysis"],
            MINIO["paths"]["outputs"] + "a1_missing_analysis.csv",
        )

        # Analysis 2: Univariate Analysis
        univariate = univariate_analysis(df)
        results["analyses"]["univariate_count"] = len(univariate)
        upload_csv(univariate, MINIO["paths"]["outputs"] + "a2_univariate_analysis.csv")

        # Analysis 3: Bivariate Analysis
        bivariate = bivariate_analysis(df)
        results["analyses"]["bivariate_count"] = len(bivariate)
        upload_csv(bivariate, MINIO["paths"]["outputs"] + "a3_bivariate_analysis.csv")

        # Print summary report
        print("\n" + "-" * 35)
        print("ANALYTICS SUMMARY REPORT")
        print("-" * 35)
        ov = overview["overview"]
        print("\nDATASET OVERVIEW:")
        print(f"  Total Records: {ov['n_rows']:,}")
        print(f"  Total Features: {ov['n_columns']}")
        print(f"  Numeric Features: {ov['n_numeric']}")
        print(f"  Categorical Features: {ov['n_categorical']}")
        print(f"  Memory Usage: {ov['memory_mb']:.2f} MB")

        if "default_rate" in ov:
            print("\nTARGET DISTRIBUTION:")
            print(f"  Default Rate: {ov['default_rate']:.2%}")
            print(f"  Defaults: {ov['default_count']:,}")
            print(f"  Non-Defaults: {ov['non_default_count']:,}")

        print("\nDATA QUALITY:")
        print(f"  Missing Values: {ov['missing_pct']:.2f}%")
        print(f"  Duplicate Rows: {ov['duplicate_rows']}")

        # Top missing features
        top_missing = overview["missing_analysis"].head(5)
        if top_missing["missing_pct"].max() > 0:
            print("\nTOP MISSING FEATURES:")
            for _, row in top_missing.iterrows():
                if row["missing_pct"] > 0:
                    print(f"  {row['column']}: {row['missing_pct']:.1f}%")

        # Features with high discriminative power
        if len(bivariate) > 0 and "default_rate_range" in bivariate.columns:
            top_discrim = bivariate.nlargest(5, "default_rate_range")
            print("\nTOP DISCRIMINATIVE FEATURES:")
            for _, row in top_discrim.iterrows():
                if pd.notna(row.get("default_rate_range")):
                    print(f"  {row['feature']}: DR range {row['default_rate_range']:.2%}")

        results["status"] = "success"

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)

    # Final timing
    pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
    results["completed_at"] = datetime.now().isoformat()
    results["total_duration"] = pipeline_duration

    print("-" * 70)
    print(f"Total Duration: {pipeline_duration:.2f}s")
    print(f"Final Status: {results['status'].upper()}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Analytics Pipeline")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing analyses only",
    )

    args = parser.parse_args()

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results = run_analytics_pipeline(report_only=args.report_only)

    if results["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
