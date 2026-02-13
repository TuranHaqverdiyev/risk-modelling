from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from src.utils.io import get_s3_client as _get_s3_client, load_config as _load_config


st.set_page_config(page_title="Cutoff Simulator", layout="wide")

st.header("🎯 Cut-off & Approval Simulator")


# Load configs
@st.cache_resource
def load_config():
    storage = _load_config()
    with open("configs/costs.yaml") as f:
        costs = yaml.safe_load(f)
    return storage, costs


@st.cache_resource
def get_s3_client():
    storage, _ = load_config()
    return _get_s3_client(storage)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_csv_from_minio(key: str) -> pd.DataFrame:
    """Load CSV from MinIO with caching."""
    try:
        storage, _ = load_config()
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=storage["minio"]["bucket"], Key=key)
        return pd.read_csv(BytesIO(obj["Body"].read()))
    except Exception as e:
        st.error(f"Could not load {key}: {e}")
        return pd.DataFrame()  # Return empty DataFrame for error cases


def calculate_cutoff_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cutoff: float,
    fn_cost: float,
    fp_cost: float,
    lgd: float,
    avg_ead: float,
) -> dict:
    """
    Calculate all metrics for a given PD cutoff.

    Cutoff logic:
    - Predict default (y_pred=1) if P(default) >= cutoff → REJECT
    - Predict no default (y_pred=0) if P(default) < cutoff → APPROVE

    Approved population = those with predicted 0 (below cutoff)
    """
    y_pred = (y_prob >= cutoff).astype(int)

    # Confusion matrix elements
    # TN: Actual 0, Predicted 0 → Good applicants approved ✓
    # FN: Actual 1, Predicted 0 → Bad applicants approved (miss!) ✗
    # FP: Actual 0, Predicted 1 → Good applicants rejected ✗
    # TP: Actual 1, Predicted 1 → Bad applicants rejected ✓

    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()  # Defaults among approved
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()

    n_total = len(y_true)
    n_approved = tn + fn  # All predicted as 0 (good)
    n_rejected = tp + fp  # All predicted as 1 (bad)

    # Key metrics
    approval_rate = n_approved / n_total if n_total > 0 else 0
    bad_rate = fn / n_approved if n_approved > 0 else 0  # Default rate among approved

    # Expected loss (among approved loans)
    # Loss = Number of defaults * LGD * Average EAD
    expected_loss = fn * lgd * avg_ead
    expected_loss_rate = expected_loss / (n_approved * avg_ead) if n_approved > 0 else 0

    # Cost calculation
    # FN cost: Approved applicants who default
    # FP cost: Rejected applicants who wouldn't default (opportunity cost)
    total_cost = fn * fn_cost + fp * fp_cost

    # Precision, Recall for default class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "cutoff": cutoff,
        "approval_rate": approval_rate * 100,
        "rejection_rate": (1 - approval_rate) * 100,
        "bad_rate": bad_rate * 100,
        "n_approved": n_approved,
        "n_rejected": n_rejected,
        "n_bad_approved": fn,
        "n_good_rejected": fp,
        "expected_loss": expected_loss,
        "expected_loss_rate": expected_loss_rate * 100,
        "total_cost": total_cost,
        "precision": precision * 100,
        "recall": recall * 100,
        "tn": tn,
        "fn": fn,
        "fp": fp,
        "tp": tp,
    }


# Load cost config
storage_cfg, costs_cfg = load_config()
paths = storage_cfg["minio"]["paths"]

# Load cutoff analysis results or predictions
cutoff_results_key = paths["outputs"] + "ds5_cutoff_analysis.csv"
df_cutoff = load_csv_from_minio(cutoff_results_key)

# Also try to load test predictions for interactive simulation
test_woe_key = paths["processed"] + "test_woe.csv"
df_test = load_csv_from_minio(test_woe_key)

# Sidebar: Cost parameters
st.sidebar.header("Cost Parameters")
fn_cost = st.sidebar.number_input(
    "False Negative Cost (Default not caught)",
    min_value=1.0,
    value=float(costs_cfg.get("false_negative", 10)),
    step=1.0,
    help="Cost of approving an applicant who will default",
)
fp_cost = st.sidebar.number_input(
    "False Positive Cost (Good rejected)",
    min_value=0.1,
    value=float(costs_cfg.get("false_positive", 1)),
    step=0.1,
    help="Opportunity cost of rejecting a good applicant",
)
lgd = st.sidebar.slider(
    "Loss Given Default (LGD)",
    min_value=0.0,
    max_value=1.0,
    value=costs_cfg.get("lgd", 0.45),
    step=0.05,
    help="Expected loss percentage if a loan defaults",
)
avg_ead = st.sidebar.number_input(
    "Average Exposure at Default ($)",
    min_value=1000,
    value=costs_cfg.get("avg_ead", 50000),
    step=1000,
    help="Average loan amount",
)

# Main content
tabs = st.tabs(["📈 Cutoff Analysis", "🔄 Interactive Simulator", "📊 Pre-computed Results"])

# TAB 1: Cutoff Analysis
with tabs[0]:
    st.subheader("Cutoff Trade-off Analysis")

    if df_cutoff is not None:
        # Visualize the cutoff analysis results
        col1, col2 = st.columns(2)

        # --- Model selection ---
        summary_key = paths["outputs"] + "ds3_lr_summary.csv"
        df_summary = load_csv_from_minio(summary_key)
        model_options = (
            [row["model"] for _, row in df_summary.iterrows()]
            if df_summary is not None
            else ["Default"]
        )
        selected_model = st.selectbox(
            "Select model for cutoff analysis",
            model_options,
            index=0,
            help="Choose which model's cutoff analysis to display.",
        )

        # Load actual predictions for selected model
        pred_key = paths["outputs"] + f"ds3_lr_predictions_{selected_model}.csv"
        df_pred = load_csv_from_minio(pred_key)
        if df_pred is not None and not df_pred.empty:
            y_true = np.array(df_pred["y_true"].to_numpy(), dtype=float)
            y_prob = np.array(df_pred["y_prob"].to_numpy(), dtype=float)
            cutoffs = np.arange(0.05, 1.01, 0.05)
            results = []
            for cutoff in cutoffs:
                metrics = calculate_cutoff_metrics(
                    y_true,
                    y_prob,
                    float(cutoff),
                    float(fn_cost),
                    float(fp_cost),
                    float(lgd),
                    float(avg_ead),
                )
                results.append(metrics)
            df_cutoff_model = pd.DataFrame(results)
            # Show charts and table
            col1, col2 = st.columns(2)
            with col1:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(
                        x=df_cutoff_model["cutoff"],
                        y=df_cutoff_model["approval_rate"],
                        name="Approval Rate %",
                        line=dict(color="#3498db"),
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_cutoff_model["cutoff"],
                        y=df_cutoff_model["bad_rate"],
                        name="Bad Rate %",
                        line=dict(color="#e74c3c"),
                    ),
                    secondary_y=True,
                )
                fig.update_layout(
                    title=f"Approval Rate vs Bad Rate by Cutoff ({selected_model})",
                    xaxis_title="PD Cutoff",
                )
                fig.update_yaxes(title_text="Approval Rate %", secondary_y=False)
                fig.update_yaxes(title_text="Bad Rate %", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.line(
                    x=df_cutoff_model["cutoff"],
                    y=df_cutoff_model["expected_loss"],
                    title=f"Expected Loss by Cutoff ({selected_model})",
                    labels={"x": "PD Cutoff", "y": "Expected Loss ($)"},
                )
                st.plotly_chart(fig, use_container_width=True)
            st.subheader(f"Cutoff Metrics Table ({selected_model})")
            display_cols = [
                "cutoff",
                "approval_rate",
                "bad_rate",
                "expected_loss",
                "n_approved",
                "precision",
                "recall",
            ]
            display_cols = [c for c in display_cols if c in df_cutoff_model.columns]
            st.dataframe(
                df_cutoff_model[display_cols].style.format(
                    {
                        "cutoff": "{:.2f}",
                        "approval_rate": "{:.2f}%",
                        "bad_rate": "{:.2f}%",
                        "expected_loss": "{:,.0f}",
                        "precision": "{:.2f}%",
                        "recall": "{:.2f}%",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.warning(
                f"Predictions file not found for model '{selected_model}'. Please run the DS pipeline or check MinIO outputs."
            )
            st.info(
                "No charts or metrics available for this model until predictions are generated."
            )

# TAB 2: Interactive Simulator
with tabs[1]:
    st.subheader("Interactive Cutoff Simulator")

    st.info("""
    **How to interpret:**
    - **Cutoff**: PD threshold. Applications with P(default) >= cutoff are **rejected**
    - **Approval Rate**: Percentage of applications approved (below cutoff)
    - **Bad Rate**: Default rate among approved applications
    - **Expected Loss**: Estimated monetary loss from defaults
    """)

    # --- Model selection ---
    # Load model summary from the same location as in model_performance.py
    summary_key = paths["outputs"] + "ds3_lr_summary.csv"
    df_summary = load_csv_from_minio(summary_key)
    model_options = (
        [row["model"] for _, row in df_summary.iterrows()]
        if df_summary is not None
        else ["Default"]
    )
    selected_model = st.selectbox(
        "Select model for simulation",
        model_options,
        index=0,
        help="Choose which model's predictions to use for cutoff simulation.",
    )

    # Single cutoff selector
    cutoff = st.slider(
        "Select PD Cutoff",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        help="Applications with P(default) >= cutoff will be rejected",
    )

    if df_test is not None and "default_90p_12m" in df_test.columns:
        y_true = df_test["default_90p_12m"].to_numpy()

        # ── Load real model predictions from MinIO (MRM-F1 fix) ──────────
        storage, _ = load_config()
        pred_key = storage["minio"]["paths"]["outputs"] + f"ds3_lr_predictions_{selected_model}.csv"
        df_pred = load_csv_from_minio(pred_key)

        if df_pred is not None and not df_pred.empty and "y_prob" in df_pred.columns:
            y_prob = df_pred["y_prob"].to_numpy()
            y_true = df_pred["y_true"].to_numpy()
        else:
            st.error(
                f"⚠️ No predictions found for model '{selected_model}' in MinIO. "
                "Please run the DS pipeline (`python pipelines/run_data_science.py`) first. "
                "The cutoff simulator requires real model predictions — synthetic fallbacks "
                "have been removed to prevent label leakage (MRM-F1)."
            )
            st.stop()
        y_prob = np.clip(y_prob, 0.01, 0.99)

        metrics = calculate_cutoff_metrics(
            np.array(y_true, dtype=float),
            np.array(y_prob, dtype=float),
            float(cutoff),
            float(fn_cost),
            float(fp_cost),
            float(lgd),
            float(avg_ead),
        )

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Approval Rate", f"{metrics['approval_rate']:.1f}%")
        col2.metric("Bad Rate", f"{metrics['bad_rate']:.2f}%")
        col3.metric("Expected Loss", f"${metrics['expected_loss']:,.0f}")
        col4.metric("Total Cost", f"{metrics['total_cost']:,.0f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Approved", f"{metrics['n_approved']:,}")
        col6.metric("Rejected", f"{metrics['n_rejected']:,}")
        col7.metric("Bad Approved", f"{metrics['n_bad_approved']:,}")
        col8.metric("Good Rejected", f"{metrics['n_good_rejected']:,}")

        # Confusion matrix visualization
        st.subheader("Confusion Matrix")
        fig = go.Figure(
            data=go.Heatmap(
                z=[[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
                x=["Predicted Good (Approve)", "Predicted Bad (Reject)"],
                y=["Actual Good", "Actual Bad"],
                text=[
                    [f"TN: {metrics['tn']}", f"FP: {metrics['fp']}"],
                    [f"FN: {metrics['fn']}", f"TP: {metrics['tp']}"],
                ],
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig.update_layout(
            title=f"Confusion Matrix at Cutoff = {cutoff:.2f}",
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Generate cutoff curve
        st.subheader("Cutoff Impact Curves")
        cutoffs = np.arange(0.01, 0.51, 0.01)
        results = []
        for c in cutoffs:
            m = calculate_cutoff_metrics(
                np.array(y_true, dtype=float),
                np.array(y_prob, dtype=float),
                float(c),
                float(fn_cost),
                float(fp_cost),
                float(lgd),
                float(avg_ead),
            )
            results.append(m)

        df_results = pd.DataFrame(results)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Approval Rate vs Bad Rate", "Cost Analysis"),
        )

        fig.add_trace(
            go.Scatter(
                x=df_results["cutoff"],
                y=df_results["approval_rate"],
                name="Approval %",
                line=dict(color="#3498db"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_results["cutoff"],
                y=df_results["bad_rate"],
                name="Bad %",
                line=dict(color="#e74c3c"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_results["cutoff"],
                y=df_results["total_cost"],
                name="Total Cost",
                line=dict(color="#9b59b6"),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df_results["cutoff"],
                y=df_results["expected_loss"],
                name="Expected Loss",
                line=dict(color="#e67e22"),
            ),
            row=1,
            col=2,
        )

        # Add vertical line for selected cutoff
        fig.add_vline(x=cutoff, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_vline(x=cutoff, line_dash="dash", line_color="gray", row=1, col=2)

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Test data not available for interactive simulation. Run the DS pipeline first.")

# TAB 3: Pre-computed Results
with tabs[2]:
    st.subheader("Pre-computed Results from Pipeline")

    if df_cutoff is not None:
        # Find optimal cutoff
        if "total_cost" in df_cutoff.columns:
            optimal_idx = df_cutoff["total_cost"].idxmin()
            optimal = df_cutoff.loc[optimal_idx]

            st.success(f"**Optimal Cutoff: {optimal['cutoff']:.2f}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Approval Rate", f"{optimal.get('approval_rate', 0):.1f}%")
            col2.metric("Bad Rate", f"{optimal.get('bad_rate', 0):.2f}%")
            col3.metric("Total Cost", f"{optimal.get('total_cost', 0):,.0f}")

        # Download button
        csv_buffer = df_cutoff.to_csv(index=False)
        st.download_button(
            label="Download Cutoff Analysis CSV",
            data=csv_buffer,
            file_name="cutoff_analysis.csv",
            mime="text/csv",
        )
    else:
        st.warning("No pre-computed results available.")
