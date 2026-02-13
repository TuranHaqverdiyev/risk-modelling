import argparse

from src.utils.io import get_s3_client, load_config


cfg = load_config()

s3 = get_s3_client(cfg)

parser = argparse.ArgumentParser(description="Upload raw dataset to MinIO")
parser.add_argument(
    "--file",
    default="data/raw/dataset.xlsx",
    help="Path to the dataset file (default: data/raw/dataset.xlsx)",
)
args = parser.parse_args()

local_file = args.file
bucket = cfg["minio"]["bucket"]
remote_key = f"{cfg['minio']['paths']['raw']}dataset.xlsx"


print(f"Uploading {local_file} to {bucket}/{remote_key}...")
s3.upload_file(local_file, bucket, remote_key)
print("Done.")
