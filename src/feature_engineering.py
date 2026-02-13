"""
Feature Engineering Module for Credit Risk Model
1. Information Value (IV) calculation for feature selection
2. Correlation-based filtering (keep feature with higher IV)
3. VIF-based multicollinearity removal
4. Weight of Evidence (WoE) transformation
5. Proper serialization of binning parameters for consistent train/test application
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.utils.io import (
    get_s3_client,
    load_config,
    load_csv as _io_load_csv,
    upload_csv as _io_upload_csv,
    upload_pickle as _io_upload_pickle,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# feature selection thresholds

CORRELATION_THRESHOLD = 0.85  # Remove one of pair if |corr| > threshold
VIF_THRESHOLD = 10.0  # Remove features with VIF > threshold
VIF_MAX_ITERATIONS = 50  # Max iterations for VIF removal

# Load configs
cfg = load_config()
s3 = get_s3_client(cfg)

with open("configs/features.yaml") as f:
    feat_cfg = yaml.safe_load(f)

MINIO = cfg["minio"]
BUCKET = MINIO["bucket"]

TRAIN_KEY = MINIO["paths"]["processed"] + "train.csv"
TEST_KEY = MINIO["paths"]["processed"] + "test.csv"
TRAIN_WOE_KEY = MINIO["paths"]["processed"] + "train_woe.csv"
TEST_WOE_KEY = MINIO["paths"]["processed"] + "test_woe.csv"
WOE_BINS_KEY = MINIO["paths"]["processed"] + "woe_bins.pkl"
IV_TABLE_KEY = MINIO["paths"]["outputs"] + "ds2_iv_table.csv"

TARGET = "default_90p_12m"

# Feature selection thresholds from config
IV_MIN = feat_cfg["iv_thresholds"]["min"]
IV_MAX = feat_cfg["iv_thresholds"]["max"]
WOE_BINS = feat_cfg["woe"]["bins"]

# MinIO client


def load_csv(key: str) -> pd.DataFrame:
    """Load CSV from MinIO (delegates to utils.io)."""
    return _io_load_csv(key, cfg)


def upload_csv(df: pd.DataFrame, key: str) -> None:
    """Upload DataFrame as CSV to MinIO (delegates to utils.io)."""
    _io_upload_csv(df, key, cfg)
    logger.info("Uploaded: %s", key)


def upload_pickle(obj, key: str) -> None:
    """Upload pickle object to MinIO (delegates to utils.io)."""
    _io_upload_pickle(obj, key, cfg)
    logger.info("Uploaded pickle: %s", key)


def calc_iv(feature: pd.Series, target: pd.Series, bins: int = 10) -> float:
    """
    Calculate Information Value for a feature.
    IV interpretation:
    - < 0.02: Not useful
    - 0.02 - 0.1: Weak predictor
    - 0.1 - 0.3: Medium predictor
    - 0.3 - 0.5: Strong predictor
    - > 0.5: Suspicious
    """
    df_tmp = pd.DataFrame({"x": feature, "y": target}).dropna()

    if len(df_tmp) == 0 or df_tmp["x"].nunique() < 2:
        return 0.0

    try:
        df_tmp["bin"] = pd.qcut(df_tmp["x"], q=bins, duplicates="drop")
    except ValueError:
        return 0.0

    grouped = df_tmp.groupby("bin", observed=True)["y"]
    total_good = (target == 0).sum()
    total_bad = (target == 1).sum()

    if total_good == 0 or total_bad == 0:
        return 0.0

    dist_good = (grouped.count() - grouped.sum()) / total_good
    dist_bad = grouped.sum() / total_bad

    # Avoid log(0) with small epsilon
    eps = 1e-6
    iv = ((dist_good - dist_bad) * np.log((dist_good + eps) / (dist_bad + eps))).sum()

    return float(iv)


# multicollinearity handling functions


def remove_high_correlation(
    X: pd.DataFrame, iv_dict: Dict[str, float], threshold: float = CORRELATION_THRESHOLD
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features, keeping the one with higher IV.
    - Model stability (correlated features cause unstable coefficients)
    - Interpretability (avoid double-counting same signal)
    - Regulatory compliance (explainability requirements)
    Args:
        X: Feature matrix (numeric features only)
        iv_dict: Dictionary mapping feature names to their IV scores
        threshold: Correlation threshold (default 0.85)

    Returns:
        Tuple of (filtered DataFrame, list of removed features)
    """
    logger.info(f"Checking for highly correlated features (threshold: {threshold})...")

    # Calculate correlation matrix
    corr_matrix = X.corr().abs()

    # Get upper triangle to avoid duplicate pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find pairs with correlation > threshold
    high_corr_pairs = []
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            corr_val = upper_tri.loc[idx, col]
            if pd.notna(corr_val) and corr_val > threshold:  # type: ignore
                high_corr_pairs.append((idx, col, corr_val))

    # Track removed features
    removed_features = set()

    # For each high-correlation pair, remove the one with lower IV
    for feat1, feat2, corr_val in high_corr_pairs:
        # Skip if either already removed
        if feat1 in removed_features or feat2 in removed_features:
            continue

        iv1 = iv_dict.get(feat1, 0.0)
        iv2 = iv_dict.get(feat2, 0.0)

        # Remove feature with lower IV
        if iv1 >= iv2:
            to_remove = feat2
            to_keep = feat1
        else:
            to_remove = feat1
            to_keep = feat2

        removed_features.add(to_remove)
        logger.info(
            f"  Removed '{to_remove}' (IV={iv_dict.get(to_remove, 0):.4f}) - "
            f"correlated with '{to_keep}' (IV={iv_dict.get(to_keep, 0):.4f}, r={corr_val:.3f})"
        )

    # Filter DataFrame
    kept_features = [c for c in X.columns if c not in removed_features]
    X_filtered = X[kept_features]

    logger.info(
        f"Correlation filter: {len(X.columns)} -> {len(X_filtered.columns)} features "
        f"({len(removed_features)} removed)"
    )

    return X_filtered, list(removed_features)


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for each feature.
    VIF measures how much the variance of a coefficient is inflated
    due to multicollinearity with other features.
    VIF interpretation:
    - VIF = 1: No correlation with other features
    - VIF < 5: Low multicollinearity (acceptable)
    - VIF 5-10: Moderate multicollinearity (caution)
    - VIF > 10: High multicollinearity (should remove)
    Args:
        X: Feature matrix (numeric, no missing values)

    Returns:
        DataFrame with feature names and VIF values
    """
    from sklearn.linear_model import LinearRegression

    vif_data = []
    X_arr = X.to_numpy()

    for i, col in enumerate(X.columns):
        # Regress feature i on all other features
        y_i = X_arr[:, i]
        X_others = np.delete(X_arr, i, axis=1)

        if X_others.shape[1] == 0:
            vif_data.append({"feature": col, "vif": 1.0})
            continue

        try:
            lr = LinearRegression()
            lr.fit(X_others, y_i)
            r_squared = lr.score(X_others, y_i)

            # VIF = 1 / (1 - R^2)
            if r_squared >= 1.0:
                vif = np.inf
            else:
                vif = 1 / (1 - r_squared)

            vif_data.append({"feature": col, "vif": vif})
        except Exception:
            vif_data.append({"feature": col, "vif": np.nan})

    return pd.DataFrame(vif_data).sort_values("vif", ascending=False)


def remove_high_vif(
    X: pd.DataFrame,
    iv_dict: Dict[str, float],
    threshold: float = VIF_THRESHOLD,
    max_iterations: int = VIF_MAX_ITERATIONS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Iteratively remove features with high VIF until all VIF < threshold.
    Strategy: At each iteration, remove the feature with highest VIF
    that also has the lowest IV (preserve predictive power).
    Args:
        X: Feature matrix
        iv_dict: Dictionary of feature IVs
        threshold: VIF threshold (default 10.0)
        max_iterations: Maximum removal iterations

    Returns:
        Tuple of (filtered DataFrame, list of removed features)
    """
    logger.info(f"Checking for multicollinearity via VIF (threshold: {threshold})...")

    X_work = X.copy()
    removed_features = []

    for iteration in range(max_iterations):
        # Handle missing values for VIF calculation
        X_clean = X_work.fillna(X_work.median())

        # Calculate VIF
        vif_df = calculate_vif(X_clean)
        max_vif = vif_df["vif"].max()

        if max_vif <= threshold or np.isnan(max_vif):
            break

        # Get features with VIF > threshold
        high_vif_features = vif_df[vif_df["vif"] > threshold]["feature"].tolist()

        # Remove feature with highest VIF and lowest IV
        if len(high_vif_features) > 0:
            # Sort by IV ascending (remove lowest IV first among high VIF)
            candidates = [
                (
                    f,
                    iv_dict.get(f, 0.0),
                    vif_df[vif_df["feature"] == f]["vif"].to_numpy()[0],
                )
                for f in high_vif_features
            ]
            candidates.sort(key=lambda x: (x[1], -x[2]))  # Sort by IV asc, then VIF desc

            to_remove = candidates[0][0]
            removed_iv = candidates[0][1]
            removed_vif = candidates[0][2]

            removed_features.append(to_remove)
            X_work = X_work.drop(columns=[to_remove])

            logger.info(
                (
                    f"  Iteration {iteration + 1}: Removed '{to_remove}' "
                    f"(VIF={removed_vif:.2f}, IV={removed_iv:.4f})"
                )
            )

    logger.info(
        f"VIF filter: {len(X.columns)} -> {len(X_work.columns)} features "
        f"({len(removed_features)} removed)"
    )

    return X_work, removed_features


def comprehensive_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    iv_min: float = 0.02,
    iv_max: float = 1.0,
    corr_threshold: float = CORRELATION_THRESHOLD,
    vif_threshold: float = VIF_THRESHOLD,
) -> Tuple[pd.DataFrame, Dict]:
    """
     feature selection pipeline
    Steps:
    1. Calculate IV for all features
    2. Filter by IV (remove non-predictive and suspicious features)
    3. Remove highly correlated features (keep higher IV)
    4. Remove high VIF features (handle remaining multicollinearity)
    Args:
        X: Feature matrix
        y: Target variable
        iv_min: Minimum IV threshold
        iv_max: Maximum IV threshold
        corr_threshold: Correlation threshold
        vif_threshold: VIF threshold

    Returns:
        Tuple of (selected features DataFrame, selection report dict)
    """
    logger.info("-" * 35)
    logger.info("Comprehensive Feature Selection")
    logger.info("-" * 35)

    report = {
        "initial_features": len(X.columns),
        "iv_filtered": 0,
        "corr_removed": [],
        "vif_removed": [],
        "final_features": 0,
    }

    # Step 1: Calculate IV for all numeric features
    logger.info("\nSTEP 1: Calculating Information Values...")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    iv_dict = {}
    iv_rows = []
    for col in numeric_cols:
        try:
            iv = calc_iv(X[col], y)
            iv_dict[col] = iv
            iv_rows.append(
                {
                    "feature": col,
                    "iv": round(iv, 4),
                    "power": (
                        "Not useful"
                        if iv < 0.02
                        else (
                            "Weak"
                            if iv < 0.1
                            else ("Medium" if iv < 0.3 else "Strong" if iv < 0.5 else "Suspicious")
                        )
                    ),
                }
            )
        except Exception as e:
            logger.warning(f"Could not calculate IV for {col}: {e}")
            iv_dict[col] = 0.0

    iv_df = pd.DataFrame(iv_rows).sort_values("iv", ascending=False)

    # Step 2: Filter by IV range
    logger.info(f"\nSTEP 2: Filtering by IV range [{iv_min}, {iv_max}]...")
    iv_selected = iv_df[(iv_df["iv"] >= iv_min) & (iv_df["iv"] <= iv_max)]["feature"].tolist()

    # Also include features with object dtype (categorical) - they'll be handled separately
    # cat_cols removed (unused variable)

    # For numeric only in multicollinearity check
    X_numeric = X[iv_selected].copy()
    report["iv_filtered"] = len(iv_selected)

    logger.info(f"IV filter: {len(numeric_cols)} -> {len(iv_selected)} features")

    # Step 3: Remove highly correlated features
    logger.info(f"\nSTEP 3: Removing highly correlated features (r > {corr_threshold})...")
    X_decorr, corr_removed = remove_high_correlation(X_numeric, iv_dict, corr_threshold)
    report["corr_removed"] = corr_removed

    # Step 4: Remove high VIF features
    logger.info(f"\nSTEP 4: Removing high VIF features (VIF > {vif_threshold})...")
    X_final, vif_removed = remove_high_vif(X_decorr, iv_dict, vif_threshold)
    report["vif_removed"] = vif_removed

    report["final_features"] = len(X_final.columns)
    report["iv_table"] = iv_df
    report["iv_dict"] = iv_dict
    report["selected_features"] = X_final.columns.tolist()

    # Summary
    logger.info("\n" + "-" * 35)
    logger.info("FEATURE SELECTION SUMMARY")
    logger.info("-" * 35)
    logger.info(f"Initial features: {report['initial_features']}")
    logger.info(f"After IV filter:  {report['iv_filtered']}")
    logger.info(f"After corr filter: {len(X_decorr.columns)} (-{len(corr_removed)})")
    logger.info(f"After VIF filter:  {report['final_features']} (-{len(vif_removed)})")
    logger.info(f"Final selected:    {report['final_features']} features")

    return X_final, report


# Import WoETransformer from the consolidated transformers module
from src.transformers import WoETransformer  # noqa: E402


def main():
    """
    Main feature engineering pipeline with advanced features and best practices.

    Pipeline steps:
    1. Load and generate advanced features
    2. Comprehensive feature selection (IV + correlation + VIF)
    3. WoE transformation on selected features
    4. Save all outputs for downstream modeling
    """

    logger.info("-" * 35)
    logger.info("Feature Engineering Pipeline")
    logger.info("-" * 35)

    logger.info("\nLoading training data...")
    train_df = load_csv(TRAIN_KEY)
    test_df = load_csv(TEST_KEY)

    # STEP 1: Generate Advanced Features
    logger.info("\n" + "-" * 35)
    logger.info("STEP 1: Generating Advanced Features")
    logger.info("-" * 35)

    from src.advanced_features import generate_advanced_features

    # Generate advanced features for train and test
    train_enhanced, feature_summary = generate_advanced_features(train_df.drop(columns=[TARGET]))
    test_enhanced, _ = generate_advanced_features(test_df.drop(columns=[TARGET]))

    # Add target back
    train_enhanced[TARGET] = train_df[TARGET].to_numpy()  # type: ignore
    test_enhanced[TARGET] = test_df[TARGET].to_numpy()  # type: ignore

    logger.info(f"Train enhanced shape: {train_enhanced.shape}")  # type: ignore
    logger.info(f"Test enhanced shape: {test_enhanced.shape}")  # type: ignore

    # Save enhanced datasets (before feature selection)
    TRAIN_ENHANCED_KEY = MINIO["paths"]["processed"] + "train_enhanced.csv"
    TEST_ENHANCED_KEY = MINIO["paths"]["processed"] + "test_enhanced.csv"
    upload_csv(train_enhanced, TRAIN_ENHANCED_KEY)  # type: ignore
    upload_csv(test_enhanced, TEST_ENHANCED_KEY)  # type: ignore

    # STEP 2: Comprehensive Feature Selection (Best Practices)
    logger.info("\n" + "-" * 35)
    logger.info("STEP 2: Comprehensive Feature Selection")
    logger.info("-" * 35)

    y_train = train_enhanced[TARGET]  # type: ignore
    X_train = train_enhanced.drop(columns=[TARGET])  # type: ignore

    y_test = test_enhanced[TARGET]  # type: ignore
    X_test = test_enhanced.drop(columns=[TARGET])  # type: ignore

    # Run comprehensive feature selection
    X_selected, selection_report = comprehensive_feature_selection(
        X_train,
        y_train,
        iv_min=IV_MIN,
        iv_max=IV_MAX,
        corr_threshold=CORRELATION_THRESHOLD,
        vif_threshold=VIF_THRESHOLD,
    )

    # Get IV table and selected features
    iv_df = selection_report["iv_table"]
    selected_features = selection_report["selected_features"]

    # Upload IV table
    upload_csv(iv_df, IV_TABLE_KEY)

    # Print top features by IV
    print("\n" + "-" * 35)
    print("Top 25 Features by Information Value")
    print("-" * 35)
    print(iv_df.head(25).to_string(index=False))

    # Print selected features
    print("\n" + "-" * 35)
    print(f"Selected Features: ({len(selected_features)}) - After IV + Corr + VIF filters")
    print("-" * 35)
    for i, f in enumerate(selected_features[:30]):
        iv_val = selection_report["iv_dict"].get(f, 0)
        print(f"  {i + 1:2d}. {f:<40} IV={iv_val:.4f}")
    if len(selected_features) > 30:
        print(f"  ... and {len(selected_features) - 30} more")

    # Save selected feature list
    SELECTED_FEATURES_KEY = MINIO["paths"]["processed"] + "selected_features.pkl"
    upload_pickle(
        {
            "features": selected_features,
            "iv_dict": selection_report["iv_dict"],
            "corr_removed": selection_report["corr_removed"],
            "vif_removed": selection_report["vif_removed"],
        },
        SELECTED_FEATURES_KEY,
    )

    # Save selected features dataset (for Model 3: Advanced without WoE)
    TRAIN_SELECTED_KEY = MINIO["paths"]["processed"] + "train_selected.csv"
    TEST_SELECTED_KEY = MINIO["paths"]["processed"] + "test_selected.csv"

    train_selected = X_train[selected_features].copy()
    train_selected[TARGET] = y_train.to_numpy()
    test_selected = X_test[selected_features].copy()
    test_selected[TARGET] = y_test.to_numpy()

    upload_csv(train_selected, TRAIN_SELECTED_KEY)
    upload_csv(test_selected, TEST_SELECTED_KEY)

    if len(selected_features) == 0:
        logger.error("No features selected! Check thresholds.")
        return

    # STEP 3: WoE Transformation
    logger.info("\n" + "-" * 35)
    logger.info("STEP 3: WoE Transformation")
    logger.info("-" * 35)

    # Fit WoE transformer on training data ONLY
    logger.info("Fitting WoE transformer on training data...")
    woe_transformer = WoETransformer(bins=WOE_BINS)
    woe_transformer.fit(X_train, y_train, selected_features)

    # Transform both train and test
    logger.info("Transforming training data...")
    X_train_woe = woe_transformer.transform(X_train)
    X_train_woe[TARGET] = y_train.to_numpy()

    logger.info("Transforming test data...")
    X_test_woe = woe_transformer.transform(X_test)
    X_test_woe[TARGET] = y_test.to_numpy()

    # Upload transformed datasets
    upload_csv(X_train_woe, TRAIN_WOE_KEY)
    upload_csv(X_test_woe, TEST_WOE_KEY)

    # Save WoE transformer parameters for later use
    upload_pickle(woe_transformer.get_params(), WOE_BINS_KEY)

    # SUMMARY
    print("\n" + "-" * 35)
    print("Feature Engineering Completed - Summary")
    print("-" * 35)
    print(f"""
Pipeline Results:
-----------------
1. Advanced Features Generated:
   - Train: {train_enhanced.shape}
   - Test: {test_enhanced.shape}

2. Feature Selection (Best Practices Applied):
   - Initial features: {selection_report["initial_features"]}
   - After IV filter: {selection_report["iv_filtered"]} (IV in [{IV_MIN}, {IV_MAX}])
     - After correlation filter: \
         {selection_report["iv_filtered"] - len(selection_report["corr_removed"])} (r < {CORRELATION_THRESHOLD})
     - After VIF filter: \
         {selection_report["final_features"]} (VIF < {VIF_THRESHOLD})
         (VIF < {VIF_THRESHOLD})
   - Removed for high correlation: {len(selection_report["corr_removed"])}
   - Removed for high VIF: {len(selection_report["vif_removed"])}

3. WoE Transformation:
   - Train WoE: {X_train_woe.shape}
   - Test WoE: {X_test_woe.shape}

Files Saved to MinIO:
---------------------
  - {TRAIN_ENHANCED_KEY} (all advanced features)
  - {TEST_ENHANCED_KEY}
  - {TRAIN_SELECTED_KEY} (selected features, no WoE - for Model 3)
  - {TEST_SELECTED_KEY}
  - {TRAIN_WOE_KEY} (selected features + WoE - for Model 4)
  - {TEST_WOE_KEY}
  - {WOE_BINS_KEY} (WoE transformer)
  - {IV_TABLE_KEY} (IV scores)
  - {SELECTED_FEATURES_KEY} (selection metadata)
""")


if __name__ == "__main__":
    main()
