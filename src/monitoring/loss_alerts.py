"""
Loss Monitoring and Alert System
This module monitors actual vs expected losses and generates alerts when:
1. Actual default rate exceeds expected rate
2. Loss amounts exceed thresholds
3. Model decisions diverge from expected behavior
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.utils.io import get_s3_client, load_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configs

cfg = load_config()

with open("configs/costs.yaml") as f:
    costs_cfg = yaml.safe_load(f)

MINIO = cfg["minio"]
BUCKET = MINIO["bucket"]

s3 = get_s3_client(cfg)


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """Represents a monitoring alert."""

    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    expected_value: float
    actual_value: float
    threshold: float
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "metric_name": self.metric_name,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


class LossMonitor:
    """
    Monitors portfolio losses and generates alerts.
    """

    def __init__(self):
        self.alerts: List[Alert] = []
        self.lgd = costs_cfg["loss"]["lgd"]  # Loss Given Default
        self.avg_ead = costs_cfg["loss"]["avg_ead"]  # Average Exposure at Default

        # Alert thresholds (can be moved to config)
        self.thresholds = {
            "default_rate_relative": 1.2,  # Alert if actual DR > 1.2x expected
            "default_rate_absolute": 0.15,  # Alert if DR > 15%
            "loss_relative": 1.3,  # Alert if actual loss > 1.3x expected
            "approval_rate_min": 0.3,  # Alert if approval rate < 30%
            "approval_rate_max": 0.9,  # Alert if approval rate > 90%
        }

    def add_alert(
        self,
        severity: AlertSeverity,
        title: str,
        description: str,
        metric_name: str,
        expected: float,
        actual: float,
        threshold: float,
    ):
        """Add an alert to the list."""
        alert = Alert(
            severity=severity,
            title=title,
            description=description,
            metric_name=metric_name,
            expected_value=expected,
            actual_value=actual,
            threshold=threshold,
            timestamp=datetime.now().isoformat(),
        )
        self.alerts.append(alert)
        logger.log(
            (
                logging.WARNING
                if severity == AlertSeverity.WARNING
                else (logging.ERROR if severity == AlertSeverity.CRITICAL else logging.INFO)
            ),
            f"[{severity.value}] {title}: {description}",
        )

    def check_default_rate(self, expected_rate: float, actual_rate: float, sample_size: int):
        """Check if default rate is within acceptable bounds."""

        # Relative check
        if actual_rate > expected_rate * self.thresholds["default_rate_relative"]:
            self.add_alert(
                AlertSeverity.CRITICAL,
                "High Default Rate",
                f"Actual default rate ({actual_rate:.2%}) exceeds "
                f"{self.thresholds['default_rate_relative']:.0%} of expected ({expected_rate:.2%})",
                "default_rate_relative",
                expected_rate,
                actual_rate,
                expected_rate * self.thresholds["default_rate_relative"],
            )
        elif actual_rate > expected_rate * 1.1:
            self.add_alert(
                AlertSeverity.WARNING,
                "Elevated Default Rate",
                (
                    f"Actual default rate ({actual_rate:.2%}) is 10%+ above expected "
                    f"({expected_rate:.2%})"
                ),
                "default_rate_elevated",
                expected_rate,
                actual_rate,
                expected_rate * 1.1,
            )

        # Absolute check
        if actual_rate > self.thresholds["default_rate_absolute"]:
            self.add_alert(
                AlertSeverity.CRITICAL,
                "Default Rate Threshold Exceeded",
                (
                    f"Default rate ({actual_rate:.2%}) exceeds absolute threshold "
                    f"({self.thresholds['default_rate_absolute']:.0%})"
                ),
                "default_rate_absolute",
                self.thresholds["default_rate_absolute"],
                actual_rate,
                self.thresholds["default_rate_absolute"],
            )

    def check_approval_rate(self, approval_rate: float, sample_size: int):
        """Check if approval rate is within acceptable bounds."""

        if approval_rate < self.thresholds["approval_rate_min"]:
            self.add_alert(
                AlertSeverity.WARNING,
                "Low Approval Rate",
                f"Approval rate ({approval_rate:.2%}) below minimum "
                f"({self.thresholds['approval_rate_min']:.0%}). "
                f"Model may be too conservative.",
                "approval_rate_low",
                self.thresholds["approval_rate_min"],
                approval_rate,
                self.thresholds["approval_rate_min"],
            )

        if approval_rate > self.thresholds["approval_rate_max"]:
            self.add_alert(
                AlertSeverity.WARNING,
                "High Approval Rate",
                f"Approval rate ({approval_rate:.2%}) above maximum "
                f"({self.thresholds['approval_rate_max']:.0%}). "
                f"Model may be too permissive.",
                "approval_rate_high",
                self.thresholds["approval_rate_max"],
                approval_rate,
                self.thresholds["approval_rate_max"],
            )

    def check_portfolio_loss(self, expected_loss: float, actual_loss: float):
        """Check if portfolio loss is within acceptable bounds."""

        if actual_loss > expected_loss * self.thresholds["loss_relative"]:
            self.add_alert(
                AlertSeverity.CRITICAL,
                "High Portfolio Loss",
                f"Actual loss ({actual_loss:,.0f}) exceeds "
                f"{self.thresholds['loss_relative']:.0%} of expected ({expected_loss:,.0f})",
                "portfolio_loss",
                expected_loss,
                actual_loss,
                expected_loss * self.thresholds["loss_relative"],
            )
        elif actual_loss > expected_loss * 1.1:
            self.add_alert(
                AlertSeverity.WARNING,
                "Elevated Portfolio Loss",
                f"Actual loss ({actual_loss:,.0f}) is 10%+ above expected ({expected_loss:,.0f})",
                "portfolio_loss_elevated",
                expected_loss,
                actual_loss,
                expected_loss * 1.1,
            )

    def check_bad_rate_among_approved(self, expected_bad_rate: float, actual_bad_rate: float):
        """Check bad rate specifically among approved applicants."""

        if actual_bad_rate > expected_bad_rate * 1.3:
            self.add_alert(
                AlertSeverity.CRITICAL,
                "High Bad Rate in Approved Portfolio",
                f"Bad rate among approved ({actual_bad_rate:.2%}) is 30%+ above "
                f"expected ({expected_bad_rate:.2%}). Model discrimination may have degraded.",
                "bad_rate_approved",
                expected_bad_rate,
                actual_bad_rate,
                expected_bad_rate * 1.3,
            )

    def calculate_expected_loss(
        self,
        n_loans: int,
        pd: float,
        lgd: Optional[float] = None,
        ead: Optional[float] = None,
    ) -> float:
        """
        Calculate Expected Loss using standard EL = PD × LGD × EAD formula.
        """
        lgd = lgd or self.lgd
        ead = ead or self.avg_ead

        return n_loans * pd * lgd * ead  # type: ignore

    def get_summary(self) -> Dict:
        """Get summary of all alerts."""
        critical = [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]
        warnings = [a for a in self.alerts if a.severity == AlertSeverity.WARNING]
        info = [a for a in self.alerts if a.severity == AlertSeverity.INFO]

        return {
            "total_alerts": len(self.alerts),
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "info_count": len(info),
            "alerts": [a.to_dict() for a in self.alerts],
        }


def load_csv(key: str) -> pd.DataFrame:
    """Load CSV from MinIO."""
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))


def upload_json(data: Dict, key: str) -> None:
    """Upload JSON to MinIO."""
    import json

    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(data, indent=2).encode("utf-8"))


def run_loss_monitoring(
    actual_data_key: Optional[str] = None,
    predictions_key: Optional[str] = None,
    target_col: str = "default_90p_12m",
    expected_default_rate: Optional[float] = None,
) -> Dict:
    """
    Run loss monitoring on actual outcomes vs predictions.

    Args:
        actual_data_key: MinIO key for actual outcomes data
        predictions_key: MinIO key for model predictions (optional)
        target_col: Target column name
        expected_default_rate: Expected portfolio default rate (if None, uses training rate)

    Returns:
        Dictionary with monitoring results and alerts
    """
    actual_data_key = actual_data_key or (MINIO["paths"]["processed"] + "test.csv")
    final_output_key = MINIO["paths"]["monitoring"] + "mon3_loss_alerts.csv"

    logger.info(f"Loading actual outcomes data: {actual_data_key}")
    df_actual = load_csv(actual_data_key)  # type: ignore

    # Get baseline expected rate from training if not provided
    if expected_default_rate is None:
        train_key = MINIO["paths"]["processed"] + "train.csv"
        df_train = load_csv(train_key)
        expected_default_rate = df_train[target_col].mean()

    # Initialize monitor
    monitor = LossMonitor()

    # Calculate actual metrics
    actual_default_rate = df_actual[target_col].mean()
    n_samples = len(df_actual)
    n_defaults = df_actual[target_col].sum()

    # Run checks
    monitor.check_default_rate(expected_default_rate, actual_default_rate, n_samples)

    # Calculate expected vs actual loss
    expected_loss = monitor.calculate_expected_loss(n_samples, expected_default_rate)
    actual_loss = monitor.calculate_expected_loss(n_samples, actual_default_rate)

    monitor.check_portfolio_loss(expected_loss, actual_loss)

    # If we have predictions, check approval-specific metrics
    if predictions_key:
        load_csv(predictions_key)
        # Add more detailed checks here

    # Get summary
    summary = monitor.get_summary()

    # Add context
    summary["monitoring_context"] = {
        "data_source": actual_data_key,
        "n_samples": n_samples,
        "n_defaults": int(n_defaults),
        "expected_default_rate": expected_default_rate,
        "actual_default_rate": actual_default_rate,
        "expected_loss": expected_loss,
        "actual_loss": actual_loss,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    upload_json(summary, final_output_key)

    # Print report
    print("\n" + "=" * 35)
    print("LOSS MONITORING & ALERT REPORT")
    print("=" * 35)
    print(f"Data Source: {actual_data_key}")
    print(f"Sample Size: {n_samples:,}")
    print(f"Actual Defaults: {int(n_defaults):,}")
    print("-" * 35)
    print("DEFAULT RATE COMPARISON:")
    print(f"  Expected: {expected_default_rate:.2%}")
    print(f"  Actual:   {actual_default_rate:.2%}")
    print(f"  Ratio:    {actual_default_rate / expected_default_rate:.2f}x")
    print("-" * 35)
    print("EXPECTED LOSS COMPARISON:")
    print(f"  Expected: {expected_loss:,.0f}")
    print(f"  Actual:   {actual_loss:,.0f}")
    print(f"  Ratio:    {actual_loss / expected_loss:.2f}x")
    print("-" * 35)
    print(f"ALERTS GENERATED: {summary['total_alerts']}")
    print(f"   Critical: {summary['critical_count']}")
    print(f"   Warning:  {summary['warning_count']}")
    print(f"   Info:     {summary['info_count']}")

    if summary["critical_count"] > 0:
        print("\n  CRITICAL ALERTS:")
        for alert in summary["alerts"]:
            if alert["severity"] == "CRITICAL":
                print(f"  - {alert['title']}: {alert['description']}")

    if summary["warning_count"] > 0:
        print("\n  WARNING ALERTS:")
        for alert in summary["alerts"]:
            if alert["severity"] == "WARNING":
                print(f"  - {alert['title']}: {alert['description']}")

    if summary["total_alerts"] == 0:
        print("\n No alerts generated - all metrics within acceptable ranges")

    print("-" * 35)
    print(f"Results saved to: {final_output_key}")
    print("=" * 35)

    return summary


if __name__ == "__main__":
    run_loss_monitoring()
