"""
Shared I/O utilities for MinIO and configuration management.

Centralises MinIO client creation, config loading (with env-var resolution),
and CSV/pickle upload/download so that every DS module doesn't reinvent them.
"""

import os
import pickle
import re
from io import BytesIO
from typing import Any, Optional

import boto3
import pandas as pd
import yaml


def _resolve_env_vars(value: str) -> str:
    """Resolve ${VAR:-default} patterns in a string using os.environ."""
    pattern = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)

    return pattern.sub(_replace, value)


def _walk_and_resolve(obj: Any) -> Any:
    """Recursively resolve env-var placeholders in a nested dict/list."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_resolve(item) for item in obj]
    return obj


def load_config(path: str = "configs/storage.yaml") -> dict:
    """Load storage config with environment-variable resolution.

    The YAML may contain ``${VAR:-default}`` placeholders which are expanded
    from ``os.environ`` at load time.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return _walk_and_resolve(raw)


def get_s3_client(cfg: Optional[dict] = None) -> boto3.client:
    """Return a configured boto3 S3 client (pointing at MinIO)."""
    if cfg is None:
        cfg = load_config()
    minio = cfg["minio"]
    return boto3.client(
        "s3",
        endpoint_url=minio["endpoint"],
        aws_access_key_id=minio["access_key"],
        aws_secret_access_key=minio["secret_key"],
    )


def load_csv(key: str, cfg: Optional[dict] = None) -> pd.DataFrame:
    """Download a CSV from MinIO and return as DataFrame."""
    if cfg is None:
        cfg = load_config()
    s3 = get_s3_client(cfg)
    obj = s3.get_object(Bucket=cfg["minio"]["bucket"], Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))


def upload_csv(df: pd.DataFrame, key: str, cfg: Optional[dict] = None) -> None:
    """Upload a DataFrame as CSV to MinIO."""
    if cfg is None:
        cfg = load_config()
    s3 = get_s3_client(cfg)
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=cfg["minio"]["bucket"], Key=key, Body=buf.getvalue())


def load_pickle(key: str, cfg: Optional[dict] = None) -> Any:
    """Download a pickle object from MinIO."""
    if cfg is None:
        cfg = load_config()
    s3 = get_s3_client(cfg)
    obj = s3.get_object(Bucket=cfg["minio"]["bucket"], Key=key)
    return pickle.loads(obj["Body"].read())  # noqa: S301


def upload_pickle(obj: Any, key: str, cfg: Optional[dict] = None) -> None:
    """Upload a picklable object to MinIO."""
    if cfg is None:
        cfg = load_config()
    s3 = get_s3_client(cfg)
    buf = BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    s3.put_object(Bucket=cfg["minio"]["bucket"], Key=key, Body=buf.getvalue())


def upload_json_bytes(data: bytes, key: str, cfg: Optional[dict] = None) -> None:
    """Upload raw bytes (e.g. JSON-encoded) to MinIO."""
    if cfg is None:
        cfg = load_config()
    s3 = get_s3_client(cfg)
    s3.put_object(Bucket=cfg["minio"]["bucket"], Key=key, Body=data)
