# Credit Modelling

[![CI/CD](https://github.com/TuranHaqverdiyev/risk-modelling/actions/workflows/ci.yml/badge.svg)](https://github.com/TuranHaqverdiyev/risk-modelling/actions/workflows/ci.yml)

End-to-End Credit Risk Modelling Project


## Features

-  **Modern ML Stack**: Built with Python 3.10+, Scikit-learn, and [uv](https://github.com/astral-sh/uv) for dependency management.
-  **Interactive Dashboard**: Streamlit-powered UI for model monitoring and analytics.
-  **Full Pipeline**: Automated end-to-end ML pipeline from data ingestion to model interpretation.
-  **Containerized**: Easy deployment using Docker and Docker Compose.
-  **CI/CD**: Automated linting and formatting via GitHub Actions.

## Project Structure

```
├── app/                    # Streamlit dashboard
│   ├── streamlit_app.py    # Main entrypoint
│   └── pages/              # Multi-page Streamlit pages
├── src/                    # Core ML pipeline modules
│   ├── utils/              # Shared I/O utilities (MinIO, config)
│   │   └── io.py
│   ├── transformers.py     # WoE transformers (shared across modules)
│   ├── train_lr.py         # Logistic regression training (4 models)
│   ├── feature_engineering.py
│   ├── advanced_features.py
│   ├── cutoff_analysis.py
│   ├── feature_importance.py
│   ├── interpretation.py
│   ├── summary.py
│   ├── split.py
│   └── upload_raw.py
├── configs/                # YAML configuration files
├── pipelines/              # Pipeline orchestration
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Multi-service orchestration
├── pyproject.toml          # Package metadata (enables clean imports)
└── requirements.txt        # Python dependencies
```

## Development

This project uses [uv](https://github.com/astral-sh/uv) for Python package and project management.

### Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and setup virtual environment
uv sync --frozen

# Activate the environment (optional, uv run handles this automatically)
source .venv/bin/activate
```

### Common Commands

| Task | Command |
|------|---------|
| **Run Streamlit App** | `uv run streamlit run app/streamlit_app.py` |
| **Run Full Pipeline** | `uv run python pipelines/run_full_pipeline.py` |
| **Add dependency** | `uv add <package>` |
| **Sync environment** | `uv sync` |
| **Lint Code** | `uv run ruff check .` |
| **Format Code** | `uv run ruff format .` |

### Docker

```bash
# Start all services (Streamlit + MinIO + MLflow)
docker compose up --build
```

## Usage

### 1. Model Pipeline Orchestration

#### Local Execution (Development)
```bash
# Execute full E2E pipeline
uv run python pipelines/run_full_pipeline.py

# Execute Data Science phase only (skip analytics and monitoring)
uv run python pipelines/run_full_pipeline.py --quick
```

#### Containerized Execution
```bash
# Execute pipeline within the application container
docker exec -it credit-risk-app python pipelines/run_full_pipeline.py
```

### 2. Analytics & Monitoring Interface

```bash
uv run streamlit run app/streamlit_app.py
```

| Module | Functionality |
| :---   | :---          |
| **Cutoff Simulator** | Scenario analysis for credit score thresholds and business impact. |
| **Model Performance**| Performance metrics (Gini, ROC-AUC, KS) across model segments.     |
| **Drift Monitoring** | Population Stability Index (PSI) and characteristic drift tracking.|

### 3. Experiment Lifecycle Management

Experiment tracking and artifact versioning are managed via **MLflow**.

- **Tracking Server**: `http://141.148.201.223:5001`
- **Artifacts**: Models, transformers, and evaluation plots are versioned per run.

---

## Infrastructure & Service Access

The platform orchestrates several integrated services via Docker Compose:

| Service | Endpoint | Primary Function |
| :--- | :--- | :--- |
| **Streamlit** | [http://141.148.201.223:8503](http://141.148.201.223:8503) | Streamlit dashboard |
| **MinIO** | [http://141.148.201.223:9031](http://141.148.201.223:9031) | MinIO Object Storage |
| **MLflow** | [http://141.148.201.223:5001](http://141.148.201.223:5001) | MLflow tracking and model registry |

