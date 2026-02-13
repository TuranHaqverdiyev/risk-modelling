"""
Stratified Train / Test Split
Loads the raw dataset from MinIO, performs a stratified split, and uploads
the resulting train.csv and test.csv back to MinIO.
"""

from io import BytesIO

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import get_s3_client, load_config, upload_csv


# Config
cfg = load_config()
paths = cfg["minio"]["paths"]

RAW_KEY = paths["raw"] + "dataset.xlsx"
TRAIN_KEY = paths["processed"] + "train.csv"
TEST_KEY = paths["processed"] + "test.csv"

TARGET_COL = "default_90p_12m"
TEST_SIZE = 0.25
RANDOM_STATE = 57

# Load raw data
s3 = get_s3_client(cfg)
obj = s3.get_object(Bucket=cfg["minio"]["bucket"], Key=RAW_KEY)
df = pd.read_excel(BytesIO(obj["Body"].read()))

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

# Train / test split
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

train_df = X_train.copy()
train_df[TARGET_COL] = y_train

test_df = X_test.copy()
test_df[TARGET_COL] = y_test

# Upload
upload_csv(train_df, TRAIN_KEY, cfg)
upload_csv(test_df, TEST_KEY, cfg)

# Sanity checks
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("Train default rate:", round(train_df[TARGET_COL].mean(), 4))
print("Test default rate :", round(test_df[TARGET_COL].mean(), 4))
