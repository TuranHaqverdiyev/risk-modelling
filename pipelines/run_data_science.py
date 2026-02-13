"""
Data Science Pipeline Orchestrator

This script runs the complete data science pipeline in the correct order:
1. Data Split (train/test)
2. Feature Engineering (IV, WoE)
3. Model Training (Logistic Regression)
4. Feature Importance Analysis
5. Cutoff Analysis
6. Model Monitoring (PSI, Performance)

Usage:
    python pipelines/run_data_science.py [--skip-split] [--skip-monitoring]
"""

import argparse
import logging
import os
import sys
from datetime import datetime


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("ds_pipeline")


class PipelineStep:
    """Represents a single pipeline step."""

    def __init__(self, name: str, module_path: str, description: str):
        self.name = name
        self.module_path = module_path
        self.description = description
        self.status = "pending"
        self.duration = None
        self.error = None

    def run(self) -> bool:
        """Execute the pipeline step."""
        logger.info(f"{'=' * 60}")
        logger.info(f"STEP: {self.name}")
        logger.info(f"Description: {self.description}")
        logger.info(f"{'=' * 60}")

        start_time = datetime.now()

        try:
            # Change to project root directory
            os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            # Import and run module
            import importlib.util

            spec = importlib.util.spec_from_file_location(self.name, self.module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module spec for {self.module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Call main() if it exists
            if hasattr(module, "main"):
                logger.info(f"Calling main() in {self.name}...")
                module.main()
            else:
                logger.info(
                    f"No main() function found in {self.name}, assuming logic ran on import."
                )

            self.status = "success"
            self.duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"SUCCESS {self.name} completed in {self.duration:.2f}s")
            return True

        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            self.duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"FAILED {self.name} failed: {e}")
            return False


def run_data_science_pipeline(
    skip_split: bool = False, skip_monitoring: bool = False, stop_on_error: bool = True
) -> dict:
    """
    Run the complete data science pipeline.

    Args:
        skip_split: Skip the train/test split step (if data already split)
        skip_monitoring: Skip monitoring steps
        stop_on_error: Stop pipeline if any step fails

    Returns:
        Dictionary with pipeline execution summary
    """

    logger.info("\n" + "=" * 70)
    logger.info("ABB CREDIT RISK - DATA SCIENCE PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Skip split: {skip_split}")
    logger.info(f"Skip monitoring: {skip_monitoring}")
    logger.info("=" * 70 + "\n")

    # Define pipeline steps
    steps = []

    # Step 1: Train/Test Split
    if not skip_split:
        steps.append(
            PipelineStep(
                name="DS1_Split",
                module_path="src/split.py",
                description="Create stratified train/test split and upload to MinIO",
            )
        )

    # Step 2: Feature Engineering
    steps.append(
        PipelineStep(
            name="DS2_FeatureEngineering",
            module_path="src/feature_engineering.py",
            description="Calculate IV, select features, apply WoE transformation",
        )
    )

    # Step 3: Model Training
    steps.append(
        PipelineStep(
            name="DS3_TrainLR",
            module_path="src/train_lr.py",
            description="Train logistic regression models and log to MLflow",
        )
    )

    # Step 4: Feature Importance
    steps.append(
        PipelineStep(
            name="DS4_FeatureImportance",
            module_path="src/feature_importance.py",
            description="Extract and analyze feature coefficients",
        )
    )

    # Step 5: Cutoff Analysis
    steps.append(
        PipelineStep(
            name="DS5_CutoffAnalysis",
            module_path="src/cutoff_analysis.py",
            description="Analyze optimal PD cutoff thresholds",
        )
    )

    # Step 6: Monitoring (optional)
    if not skip_monitoring:
        steps.append(
            PipelineStep(
                name="MON1_PSI",
                module_path="src/monitoring/psi.py",
                description="Calculate Population Stability Index",
            )
        )
        steps.append(
            PipelineStep(
                name="MON2_Performance",
                module_path="src/monitoring/performance.py",
                description="Evaluate model performance metrics",
            )
        )
        steps.append(
            PipelineStep(
                name="MON3_LossAlerts",
                module_path="src/monitoring/loss_alerts.py",
                description="Check for loss threshold breaches",
            )
        )

    # Execute steps
    pipeline_start = datetime.now()
    results = {
        "started_at": pipeline_start.isoformat(),
        "steps": [],
        "status": "running",
    }

    for step in steps:
        success = step.run()
        results["steps"].append(
            {
                "name": step.name,
                "description": step.description,
                "status": step.status,
                "duration": step.duration,
                "error": step.error,
            }
        )

        if not success and stop_on_error:
            results["status"] = "failed"
            results["failed_step"] = step.name
            break

    # Final summary
    pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
    results["completed_at"] = datetime.now().isoformat()
    results["total_duration"] = pipeline_duration

    if results["status"] != "failed":
        results["status"] = "success"

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)

    for step_result in results["steps"]:
        status_text = "PASSED" if step_result["status"] == "success" else "FAILED"
        duration = step_result["duration"] or 0
        print(f"  [{status_text}] {step_result['name']}: {step_result['status']} ({duration:.2f}s)")

    print("-" * 70)
    print(f"Total Duration: {pipeline_duration:.2f}s")
    print(f"Final Status: {results['status'].upper()}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Data Science Pipeline")
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip train/test split (use existing split)",
    )
    parser.add_argument("--skip-monitoring", action="store_true", help="Skip monitoring steps")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline even if a step fails",
    )

    args = parser.parse_args()

    results = run_data_science_pipeline(
        skip_split=args.skip_split,
        skip_monitoring=args.skip_monitoring,
        stop_on_error=not args.continue_on_error,
    )

    # Exit with error code if pipeline failed
    if results["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
