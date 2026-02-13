"""
Cut-off Analysis
- y=1 -> Default (bad customer)
- y=0 -> Non-default (good customer)
- We approve applicants predicted as good (low PD score < cutoff)
- We reject applicants predicted as bad (high PD score >= cutoff)

Key metrics:
- Approval Rate: Proportion of applicants approved
- Bad Rate: Proportion of actual defaults among approved applicants
- Expected Loss: COST_FN * FN + COST_FP * FP
"""

import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from src.utils.io import (
    get_s3_client,
    load_config,
    load_csv as _load_csv,
    upload_csv as _upload_csv,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET = "default_90p_12m"

# Load configs

cfg = load_config()

with open("configs/costs.yaml") as f:
    costs_cfg = yaml.safe_load(f)

MINIO = cfg["minio"]
BUCKET = MINIO["bucket"]

TRAIN_WOE_KEY = MINIO["paths"]["processed"] + "train_woe.csv"
TEST_WOE_KEY = MINIO["paths"]["processed"] + "test_woe.csv"
OUTPUT_KEY = MINIO["paths"]["outputs"] + "ds5_cutoff_analysis.csv"

# Load costs from centralized config
COST_FN = costs_cfg["costs"]["false_negative"]  # Cost of approving a defaulter
COST_FP = costs_cfg["costs"]["false_positive"]  # Cost of rejecting a good customer

# MinIO client
s3 = get_s3_client(cfg)

# Load data
train = _load_csv(TRAIN_WOE_KEY, cfg)
test = _load_csv(TEST_WOE_KEY, cfg)

X_tr = train.drop(columns=[TARGET]).fillna(train.median(numeric_only=True))
y_tr = train[TARGET]

X_te = test.drop(columns=[TARGET]).fillna(test.median(numeric_only=True))
y_te = test[TARGET]

# Align columns (safety)
X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)

# Train WoE LR
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_tr, y_tr)

pd_scores = model.predict_proba(X_te)[:, 1]

# Cut-off analysis
# credit risk, we approve when PD < cutoff (predict non-default)
# y_pred=1 means we predict DEFAULT (reject), y_pred=0 means predict non-default (approve)
results = []

for cutoff in np.arange(0.05, 1.01, 0.05):
    # If PD score >= cutoff, predict default (reject); else approve
    y_pred = (pd_scores >= cutoff).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

    # Confusion matrix interpretation for credit risk:
    # TN = Correctly approved non-defaulters (good decision)
    # FP = Incorrectly rejected non-defaulters (lost opportunity)
    # FN = Incorrectly approved defaulters (bad - causes losses)
    # TP = Correctly rejected defaulters (good decision)

    # Approved = those we predicted as non-defaulters (y_pred=0) = TN + FN
    n_approved = tn + fn
    approval_rate = n_approved / len(y_te)

    # Bad rate = actual defaults among approved = FN / (TN + FN)
    bad_rate = fn / max(n_approved, 1)

    # Expected loss: FN costs more (approved defaulter), FP is opportunity cost
    expected_loss = COST_FN * fn + COST_FP * fp

    # Additional useful metrics
    rejection_rate = (tp + fp) / len(y_te)
    precision = tp / max(tp + fp, 1)  # Among rejected, how many were actual defaults
    recall = tp / max(tp + fn, 1)  # Of all defaults, how many did we catch

    results.append(
        {
            "cutoff": round(cutoff, 2),
            "approval_rate": round(approval_rate, 4),
            "rejection_rate": round(rejection_rate, 4),
            "bad_rate": round(bad_rate, 4),
            "expected_loss": round(expected_loss, 1),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_approved": int(n_approved),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )

cutoff_df = pd.DataFrame(results)

# Find optimal cutoff (minimize expected loss)
optimal_idx = cutoff_df["expected_loss"].idxmin()
optimal_cutoff = cutoff_df.loc[optimal_idx, "cutoff"]

logger.info(f"Optimal cutoff: {optimal_cutoff} (minimizes expected loss)")
logger.info(
    f"At optimal cutoff - Approval Rate: {cutoff_df.loc[optimal_idx, 'approval_rate']:.2%}, "
    f"Bad Rate: {cutoff_df.loc[optimal_idx, 'bad_rate']:.2%}"
)

# Save to MinIO
_upload_csv(cutoff_df, OUTPUT_KEY, cfg)

print("\n" + "-" * 30)
print("CUTOFF ANALYSIS RESULTS")
print("-" * 30)
print(cutoff_df.to_string(index=False))
print("-" * 113)
print(f"\nOptimal Cutoff: {optimal_cutoff}")
print(f"Saved to MinIO: {OUTPUT_KEY}")
