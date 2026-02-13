"""
Pipeline Summary Module
Generates comprehensive summaries of DS pipeline outputs:
- Model performance summary
- Feature selection summary
- Cutoff recommendations
- Monitoring alerts summary
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.io import (
    get_s3_client,
    load_config,
    load_csv as _load_csv,
    upload_csv as _upload_csv,
    upload_json_bytes,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

cfg = load_config()
MINIO = cfg["minio"]
BUCKET = MINIO["bucket"]

# MinIO client
s3 = get_s3_client(cfg)


def load_csv_safe(key: str) -> Optional[pd.DataFrame]:
    """Load CSV from MinIO, return None if not found."""
    try:
        return _load_csv(key, cfg)
    except Exception as e:
        logger.warning(f"Could not load {key}: {e}")
        return None


def upload_json(data: Dict, key: str) -> None:
    """Upload dict as JSON to MinIO."""
    upload_json_bytes(json.dumps(data, indent=2, default=str).encode(), key, cfg)


def upload_csv(df: pd.DataFrame, key: str) -> None:
    """Upload DataFrame as CSV to MinIO (delegates to utils.io)."""
    _upload_csv(df, key, cfg)


def generate_model_summary(paths: Dict) -> Dict[str, Any]:
    """Generate model performance summary."""
    summary = {"generated_at": datetime.now().isoformat(), "models": []}

    # Load model summary
    df_summary = load_csv_safe(paths["outputs"] + "ds3_lr_summary.csv")

    # If CSV doesn't exist, try to get metrics from feature importance file
    if df_summary is None:
        # df_importance removed (unused variable)
        # Model metrics would need to come from MLflow in production
        pass

    if df_summary is not None:
        for _, row in df_summary.iterrows():
            model_info = {
                "model": row.get("model", "unknown"),
                "auc": float(row.get("auc", 0)),
                "ks": float(row.get("ks", 0)),
                "gini": float(row.get("gini", 0)),
                "precision": float(row.get("precision", 0)),
                "recall": float(row.get("recall", 0)),
            }
            summary["models"].append(model_info)

        # Best model
        best_idx = df_summary.get("auc", pd.Series([0])).idxmax()
        summary["best_model"] = df_summary.loc[best_idx].to_dict()

    return summary


def generate_feature_summary(paths: Dict) -> Dict[str, Any]:
    """Generate feature selection summary."""
    summary = {
        "total_features": 0,
        "selected_features": 0,
        "top_10_features": [],
        "iv_summary": {},
    }

    # Load IV values
    df_iv = load_csv_safe(paths["outputs"] + "ds2_iv_table.csv")

    if df_iv is not None:
        iv_col = (
            "iv" if "iv" in df_iv.columns else df_iv.columns[1] if len(df_iv.columns) > 1 else None
        )
        # feat_col removed (unused variable)

        if iv_col:
            summary["total_features"] = len(df_iv)
            summary["selected_features"] = (df_iv[iv_col] >= 0.02).sum()
            summary["iv_summary"] = {
                "mean": float(df_iv[iv_col].mean()),
                "max": float(df_iv[iv_col].max()),
                "min": float(df_iv[iv_col].min()),
                "features_iv_gt_0.1": int((df_iv[iv_col] >= 0.1).sum()),
                "features_iv_gt_0.3": int((df_iv[iv_col] >= 0.3).sum()),
            }

    # Load feature importance
    df_importance = load_csv_safe(paths["outputs"] + "ds4_feature_importance.csv")

    if df_importance is not None:
        top_10 = df_importance.head(10)
        summary["top_10_features"] = top_10.to_dict(orient="records")

    return summary


def generate_cutoff_summary(paths: Dict) -> Dict[str, Any]:
    """Generate cutoff analysis summary."""
    summary = {
        "recommended_cutoff": None,
        "approval_rate": None,
        "bad_rate": None,
        "cost_at_optimal": None,
        "cutoff_range_analyzed": [],
    }

    df_cutoff = load_csv_safe(paths["outputs"] + "ds5_cutoff_analysis.csv")

    if df_cutoff is not None:
        # Find optimal cutoff (minimum cost)
        if "total_cost" in df_cutoff.columns:
            optimal_idx = df_cutoff["total_cost"].idxmin()
            optimal_row = df_cutoff.loc[optimal_idx]

            summary["recommended_cutoff"] = float(optimal_row.get("cutoff", 0))  # type: ignore
            summary["approval_rate"] = float(optimal_row.get("approval_rate", 0))  # type: ignore
            summary["bad_rate"] = float(optimal_row.get("bad_rate", 0))  # type: ignore
            summary["cost_at_optimal"] = float(optimal_row.get("total_cost", 0))  # type: ignore

        summary["cutoff_range_analyzed"] = [
            float(df_cutoff["cutoff"].min()),
            float(df_cutoff["cutoff"].max()),
        ]

        # Add alternatives
        summary["alternatives"] = []
        for cutoff_val in [0.05, 0.10, 0.15, 0.20]:
            row = df_cutoff[df_cutoff["cutoff"].round(2) == cutoff_val]
            if not row.empty:
                summary["alternatives"].append(
                    {
                        "cutoff": cutoff_val,
                        "approval_rate": float(row["approval_rate"].to_numpy()[0]),
                        "bad_rate": float(row["bad_rate"].to_numpy()[0]),
                    }
                )

    return summary


def generate_data_summary(paths: Dict) -> Dict[str, Any]:
    """Generate data split summary."""
    summary = {
        "train_size": 0,
        "test_size": 0,
        "default_rate_train": 0,
        "default_rate_test": 0,
    }

    target = "default_90p_12m"

    # Load train/test data
    df_train = load_csv_safe(paths["processed"] + "train.csv")
    df_test = load_csv_safe(paths["processed"] + "test.csv")

    if df_train is not None:
        summary["train_size"] = len(df_train)
        if target in df_train.columns:
            summary["default_rate_train"] = float(df_train[target].mean())  # type: ignore

    if df_test is not None:
        summary["test_size"] = len(df_test)
        if target in df_test.columns:
            summary["default_rate_test"] = float(df_test[target].mean())  # type: ignore

    return summary


def generate_full_summary() -> Dict[str, Any]:
    """Generate complete pipeline summary."""
    paths = MINIO["paths"]

    logger.info("Generating pipeline summary...")

    summary = {
        "pipeline_summary": {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0.0",
        },
        "data": generate_data_summary(paths),
        "model_performance": generate_model_summary(paths),
        "feature_selection": generate_feature_summary(paths),
        "cutoff_analysis": generate_cutoff_summary(paths),
    }

    # Add recommendations
    summary["recommendations"] = []

    # Check model quality
    model_perf = summary["model_performance"]
    if model_perf.get("best_model"):
        auc = model_perf["best_model"].get("auc", 0)
        if auc < 0.65:
            summary["recommendations"].append(
                {
                    "type": "warning",
                    "message": (
                        f"Model AUC ({auc:.3f}) is below 0.65. "
                        "Consider feature engineering improvements."
                    ),
                }
            )
        elif auc > 0.85:
            summary["recommendations"].append(
                {
                    "type": "info",
                    "message": f"Model AUC ({auc:.3f}) is very high. Verify no data leakage.",
                }
            )

    # Check cutoff
    cutoff = summary["cutoff_analysis"].get("recommended_cutoff")
    bad_rate = summary["cutoff_analysis"].get("bad_rate")
    if cutoff and cutoff < 0.05:
        summary["recommendations"].append(
            {
                "type": "warning",
                "message": "Recommended cutoff is very low. May reject too many applicants.",
            }
        )
    if bad_rate and bad_rate > 5:
        summary["recommendations"].append(
            {
                "type": "warning",
                "message": f"Bad rate ({bad_rate:.2f}%) at optimal cutoff is high.",
            }
        )

    return summary


def main():
    """Generate and save pipeline summary."""
    paths = MINIO["paths"]

    summary = generate_full_summary()

    # Save as JSON
    json_key = paths["outputs"] + "pipeline_summary.json"
    upload_json(summary, json_key)

    # Also create a flat CSV for easy viewing
    flat_data = {"metric": [], "value": []}

    def flatten_dict(d: Dict, prefix: str = ""):
        for k, v in d.items():
            key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                flatten_dict(v, key)
            elif isinstance(v, list):
                flat_data["metric"].append(key)
                flat_data["value"].append(str(v)[:200])  # Truncate long lists
            else:
                flat_data["metric"].append(key)
                flat_data["value"].append(str(v))

    flatten_dict(summary)
    df_flat = pd.DataFrame(flat_data)
    upload_csv(df_flat, paths["outputs"] + "pipeline_summary_flat.csv")

    # Print summary
    print("\n" + "-" * 35)
    print("PIPELINE SUMMARY")
    print("-" * 35)

    data = summary["data"]
    print("\nDATA:")
    print(f"  Train size: {data['train_size']:,}")
    print(f"  Test size: {data['test_size']:,}")
    print(f"  Default rate (train): {data['default_rate_train'] * 100:.2f}%")
    print(f"  Default rate (test): {data['default_rate_test'] * 100:.2f}%")

    model = summary["model_performance"]
    if model.get("best_model"):
        print(f"\nBEST MODEL: {model['best_model'].get('model', 'N/A')}")
        print(f"  AUC: {model['best_model'].get('auc', 0):.4f}")
        print(f"  KS: {model['best_model'].get('ks', 0):.4f}")
        print(f"  Gini: {model['best_model'].get('gini', 0):.4f}")

    features = summary["feature_selection"]
    print("\nFEATURES:")
    print(f"  Total: {features['total_features']}")
    print(f"  Selected (IV>=0.02): {features['selected_features']}")

    cutoff = summary["cutoff_analysis"]
    if cutoff.get("recommended_cutoff"):
        print("\nCUTOFF RECOMMENDATION:")
        print(f"  Optimal cutoff: {cutoff['recommended_cutoff']:.2f}")
        print(f"  Approval rate: {cutoff['approval_rate']:.1f}%")
        print(f"  Bad rate: {cutoff['bad_rate']:.2f}%")

    if summary["recommendations"]:
        print("\nRECOMMENDATIONS:")
        for rec in summary["recommendations"]:
            print(f"  [{rec['type'].upper()}] {rec['message']}")

    print("-" * 35)
    print(f"Full summary saved to: {json_key}")
    print("-" * 35)


if __name__ == "__main__":
    main()
