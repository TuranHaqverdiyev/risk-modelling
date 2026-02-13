"""
WoE Transformers for Credit Risk Models

Consolidated module containing both the full WoE transformer (for the
feature-engineering pipeline) and the simple WoE transformer (for baseline
training on raw features).  Having them in one place guarantees a single
source of truth for the binning / WoE logic and avoids code duplication.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple WoE Transformer (raw features only)
# ---------------------------------------------------------------------------


class SimpleWoETransformer:
    """
    Lightweight WoE transformer intended for raw numeric features.

    Unlike :class:`WoETransformer`, this class:
    - Only works with numeric features (auto-detected)
    - Skips features with fewer than 3 unique values
    - Uses a simpler binning strategy with automatic duplicate dropping
    """

    def __init__(self, bins: int = 5):
        self.bins = bins
        self.woe_mappings: Dict[str, dict] = {}
        self.bin_edges: Dict[str, list] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleWoETransformer":
        """Learn WoE mappings from training data."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            feature = X[col].copy()

            if feature.nunique() < 3:
                continue

            try:
                binned, bin_edges = pd.qcut(
                    feature.dropna(), q=self.bins, duplicates="drop", retbins=True
                )

                df_tmp = pd.DataFrame({"bin": binned, "y": y.loc[feature.dropna().index]})
                grouped = df_tmp.groupby("bin", observed=True)["y"]

                total_good = (y == 0).sum()
                total_bad = (y == 1).sum()

                good = grouped.count() - grouped.sum()
                bad = grouped.sum()

                eps = 1e-6
                dist_good = (good + eps) / (total_good + eps)
                dist_bad = (bad + eps) / (total_bad + eps)

                woe = np.log(dist_good / dist_bad)

                self.woe_mappings[col] = woe.to_dict()
                self.bin_edges[col] = bin_edges.tolist()

            except Exception:
                continue

        logger.info("SimpleWoE fitted %d features", len(self.woe_mappings))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply WoE transformation using learned bins."""
        X_woe = pd.DataFrame(index=X.index)

        for col, woe_map in self.woe_mappings.items():
            if col not in X.columns:
                continue

            feature = X[col].copy()
            bin_edges = self.bin_edges[col]

            binned = pd.cut(feature, bins=bin_edges, include_lowest=True)

            woe_values = []
            for interval in binned:
                if pd.isna(interval):
                    woe_values.append(0.0)
                else:
                    matched = False
                    for map_interval, woe_val in woe_map.items():
                        if str(interval) == str(map_interval):
                            woe_values.append(woe_val)
                            matched = True
                            break
                    if not matched:
                        woe_values.append(0.0)

            X_woe[col + "_woe"] = woe_values

        return X_woe


# ---------------------------------------------------------------------------
# Full WoE Transformer (feature-engineering pipeline)
# ---------------------------------------------------------------------------


class WoETransformer:
    """
    Weight of Evidence Transformer with proper train/test handling.

    Learns bins and WoE mappings from training data **only**, then applies
    consistently to both train and test sets.  Supports serialisation via
    :meth:`get_params` / :meth:`from_params` for reproducible pipelines.
    """

    def __init__(self, bins: int = 5):
        self.bins = bins
        self.woe_mappings: Dict[str, dict] = {}
        self.bin_edges: Dict[str, list] = {}
        self.selected_features: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, features: list) -> "WoETransformer":
        """
        Learn WoE mappings from training data.

        Args:
            X: Training features
            y: Training target
            features: List of features to transform
        """
        self.selected_features = features

        for col in features:
            if col not in X.columns:
                logger.warning("Feature %s not in training data, skipping", col)
                continue

            feature = X[col].copy()

            try:
                binned, bin_edges = pd.qcut(feature, q=self.bins, duplicates="drop", retbins=True)

                df_tmp = pd.DataFrame({"bin": binned, "y": y})
                grouped = df_tmp.groupby("bin", observed=True)["y"]

                total_good = (y == 0).sum()
                total_bad = (y == 1).sum()

                good = grouped.count() - grouped.sum()
                bad = grouped.sum()

                dist_good = good / total_good
                dist_bad = bad / total_bad

                eps = 1e-6
                woe = np.log((dist_good + eps) / (dist_bad + eps))

                self.woe_mappings[col] = woe.to_dict()  # type: ignore
                self.bin_edges[col] = bin_edges.tolist()

                logger.info("Fitted WoE for %s: %d bins", col, len(bin_edges) - 1)

            except Exception as e:
                logger.warning("Could not fit WoE for %s: %s", col, e)
                continue

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply WoE transformation using learned bins."""
        X_woe = pd.DataFrame(index=X.index)

        for col in self.selected_features:
            if col not in self.bin_edges:
                continue

            feature = X[col].copy()
            bin_edges = self.bin_edges[col]
            woe_map = self.woe_mappings[col]

            binned = pd.cut(feature, bins=bin_edges, include_lowest=True)

            woe_values = []
            for interval in binned:
                if pd.isna(interval):
                    woe_values.append(0.0)
                else:
                    matched = False
                    for map_interval, woe_val in woe_map.items():
                        if str(interval) == str(map_interval):
                            woe_values.append(woe_val)
                            matched = True
                            break
                    if not matched:
                        woe_values.append(0.0)

            X_woe[col + "_woe"] = woe_values

        return X_woe

    def get_params(self) -> dict:
        """Return all learned parameters for serialisation."""
        return {
            "bins": self.bins,
            "woe_mappings": self.woe_mappings,
            "bin_edges": self.bin_edges,
            "selected_features": self.selected_features,
        }

    @classmethod
    def from_params(cls, params: dict) -> "WoETransformer":
        """Reconstruct transformer from serialised parameters."""
        transformer = cls(bins=params["bins"])
        transformer.woe_mappings = params["woe_mappings"]
        transformer.bin_edges = params["bin_edges"]
        transformer.selected_features = params["selected_features"]
        return transformer
