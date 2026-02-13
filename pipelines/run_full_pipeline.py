"""
Full Pipeline Orchestrator
This script runs the complete end-to-end pipeline:
1. Data Analytics (EDA & profiling)
2. Data Science (feature engineering, training, evaluation)
3. Monitoring (PSI, performance, loss alerts)

Usage:
    python pipelines/run_full_pipeline.py [options]

Options:
    --skip-analytics    Skip the analytics pipeline
    --skip-ds          Skip the data science pipeline
    --skip-monitoring  Skip the monitoring pipeline
    --quick            Run minimal version (skip analytics & monitoring)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("full_pipeline")


def print_banner():
    """Print pipeline banner."""
    banner = """
======================================================================
                     ABB CREDIT RISK - FULL PIPELINE
======================================================================
"""
    print(banner)


def run_analytics():
    """Run the analytics pipeline."""
    logger.info("=" * 70)
    logger.info("PHASE 1: DATA ANALYTICS")
    logger.info("=" * 70)

    from pipelines.run_analytics import run_analytics_pipeline

    return run_analytics_pipeline()


def run_data_science(skip_split: bool = False, skip_monitoring: bool = True):
    """Run the data science pipeline."""
    logger.info("=" * 70)
    logger.info("PHASE 2: DATA SCIENCE")
    logger.info("=" * 70)

    from pipelines.run_data_science import run_data_science_pipeline

    return run_data_science_pipeline(skip_split=skip_split, skip_monitoring=skip_monitoring)


def run_monitoring():
    """Run the monitoring pipeline."""
    logger.info("-" * 70)
    logger.info("PHASE 3: MONITORING")
    logger.info("-" * 70)

    results = {"status": "success", "components": {}}

    try:
        # PSI Monitoring
        from src.monitoring.psi import run_psi_monitoring

        run_psi_monitoring()
        results["components"]["psi"] = "success"
    except Exception as e:
        logger.error(f"PSI monitoring failed: {e}")
        results["components"]["psi"] = f"failed: {e}"

    try:
        # Performance Monitoring
        from src.monitoring.performance import run_performance_monitoring

        run_performance_monitoring()
        results["components"]["performance"] = "success"
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        results["components"]["performance"] = f"failed: {e}"

    try:
        # Loss Alerts
        from src.monitoring.loss_alerts import run_loss_monitoring

        run_loss_monitoring()
        results["components"]["loss_alerts"] = "success"
    except Exception as e:
        logger.error(f"Loss monitoring failed: {e}")
        results["components"]["loss_alerts"] = f"failed: {e}"

    # Check overall status
    if any("failed" in str(v) for v in results["components"].values()):
        results["status"] = "partial_failure"

    return results


def run_full_pipeline(
    skip_analytics: bool = False,
    skip_ds: bool = False,
    skip_monitoring: bool = False,
    quick: bool = False,
) -> dict:
    """
    Run the complete pipeline.

    Args:
        skip_analytics: Skip the analytics phase
        skip_ds: Skip the data science phase
        skip_monitoring: Skip the monitoring phase
        quick: Quick mode (skip analytics and monitoring)

    Returns:
        Dictionary with execution summary
    """

    # Quick mode settings
    if quick:
        skip_analytics = True
        skip_monitoring = True

    print_banner()

    pipeline_start = datetime.now()

    results = {
        "started_at": pipeline_start.isoformat(),
        "config": {
            "skip_analytics": skip_analytics,
            "skip_ds": skip_ds,
            "skip_monitoring": skip_monitoring,
            "quick_mode": quick,
        },
        "phases": {},
        "status": "running",
    }

    logger.info(f"Pipeline started at: {pipeline_start.isoformat()}")
    logger.info(f"Configuration: {json.dumps(results['config'], indent=2)}")

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    overall_success = True

    # Phase 1: Analytics
    if not skip_analytics:
        try:
            analytics_results = run_analytics()
            results["phases"]["analytics"] = analytics_results
            if analytics_results.get("status") == "failed":
                overall_success = False
        except Exception as e:
            logger.error(f"Analytics phase failed: {e}")
            results["phases"]["analytics"] = {"status": "failed", "error": str(e)}
            overall_success = False
    else:
        logger.info("Skipping analytics phase")
        results["phases"]["analytics"] = {"status": "skipped"}

    # Phase 2: Data Science
    if not skip_ds:
        try:
            ds_results = run_data_science(
                skip_split=False,  # Always run full DS pipeline
                skip_monitoring=True,  # Monitoring runs separately
            )
            results["phases"]["data_science"] = ds_results
            if ds_results.get("status") == "failed":
                overall_success = False
        except Exception as e:
            logger.error(f"Data science phase failed: {e}")
            results["phases"]["data_science"] = {"status": "failed", "error": str(e)}
            overall_success = False
    else:
        logger.info("Skipping data science phase")
        results["phases"]["data_science"] = {"status": "skipped"}

    # Phase 3: Monitoring
    if not skip_monitoring:
        try:
            mon_results = run_monitoring()
            results["phases"]["monitoring"] = mon_results
        except Exception as e:
            logger.error(f"Monitoring phase failed: {e}")
            results["phases"]["monitoring"] = {"status": "failed", "error": str(e)}
    else:
        logger.info("Skipping monitoring phase")
        results["phases"]["monitoring"] = {"status": "skipped"}

    # Final summary
    pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
    results["completed_at"] = datetime.now().isoformat()
    results["total_duration_seconds"] = pipeline_duration
    results["status"] = "success" if overall_success else "failed"

    # Print final summary
    print("\n" + "=" * 70)
    print("FULL PIPELINE EXECUTION SUMMARY")
    print("=" * 70)

    for phase_name, phase_result in results["phases"].items():
        status = phase_result.get("status", "unknown")
        if status == "skipped":
            icon = "SKIP"
        elif status == "success":
            icon = "SUCCESS"
        elif status == "partial_failure":
            icon = "WARNING"
        else:
            icon = "FAILED"
        print(f"  [{icon}] {phase_name.upper()}: {status}")

    print("-" * 70)
    print(f"Total Duration: {pipeline_duration:.2f} seconds")
    print(f"Final Status: {results['status'].upper()}")
    print("=" * 70)

    # Save pipeline results
    try:
        from src.utils.io import load_config, upload_json_bytes

        cfg = load_config()
        output_key = cfg["minio"]["paths"]["outputs"] + "pipeline_results.json"

        json_data = json.dumps(results, indent=2, default=str).encode("utf-8")
        upload_json_bytes(json_data, output_key, cfg)
        logger.info(f"Pipeline results saved to: {output_key}")
    except Exception as e:
        logger.warning(f"Could not save pipeline results: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Full Credit Risk Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py                    # Run everything
  python run_full_pipeline.py --quick            # Quick mode (DS only)
  python run_full_pipeline.py --skip-analytics   # Skip EDA
  python run_full_pipeline.py --skip-monitoring  # Skip monitoring
        """,
    )

    parser.add_argument(
        "--skip-analytics", action="store_true", help="Skip the analytics/EDA phase"
    )
    parser.add_argument("--skip-ds", action="store_true", help="Skip the data science phase")
    parser.add_argument("--skip-monitoring", action="store_true", help="Skip the monitoring phase")
    parser.add_argument("--quick", action="store_true", help="Quick mode: run data science only")

    args = parser.parse_args()

    results = run_full_pipeline(
        skip_analytics=args.skip_analytics,
        skip_ds=args.skip_ds,
        skip_monitoring=args.skip_monitoring,
        quick=args.quick,
    )

    if results["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
