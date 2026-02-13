"""
Model Performance Dashboard

Displays model performance metrics and interpretability insights:
- Model comparison (Raw vs WoE)
- ROC/AUC curves
- KS statistics
- Feature importance
"""

from io import BytesIO

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.io import get_s3_client as _get_s3, load_config as _load_cfg


st.set_page_config(page_title="Model Performance", layout="wide")

st.header("📊 Model Performance")


# Load configs
@st.cache_resource
def load_config():
    return _load_cfg()


@st.cache_resource
def get_s3_client():
    cfg = load_config()
    return _get_s3(cfg)


@st.cache_data(ttl=60)  # Reduce TTL to 60s for faster updates
def load_csv_from_minio(key: str) -> pd.DataFrame:
    """Load CSV from MinIO with caching."""
    try:
        cfg = load_config()
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=cfg["minio"]["bucket"], Key=key)
        return pd.read_csv(BytesIO(obj["Body"].read()))
    except Exception:
        return pd.DataFrame()  # Return empty DataFrame for error cases


@st.cache_data(ttl=300)
def get_mlflow_experiments():
    """Retrieve experiment runs from MLflow."""
    try:
        cfg = load_config()
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

        experiments = mlflow.search_experiments()
        all_runs = []
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=100)
            if isinstance(runs, pd.DataFrame) and not runs.empty:
                runs["experiment_name"] = exp.name
                all_runs.append(runs)

        if all_runs:
            return pd.concat(all_runs, ignore_index=True)
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not connect to MLflow: {e}")
        return pd.DataFrame()


# Load data
cfg = load_config()
paths = cfg["minio"]["paths"]

# Load feature importance
importance_key = paths["outputs"] + "ds4_feature_importance.csv"
df_importance = load_csv_from_minio(importance_key)

# Load model summary
summary_key = paths["outputs"] + "ds3_lr_summary.csv"
df_summary = load_csv_from_minio(summary_key)

# Load IV values from feature engineering
iv_key = paths["outputs"] + "ds2_iv_table.csv"
df_iv = load_csv_from_minio(iv_key)

# Tabs
tabs = st.tabs(
    [
        "🏆 Model Comparison",
        "📈 ROC & KS Analysis",
        "🔑 Feature Importance",
        "📋 IV Analysis",
        "🔍 MLflow Experiments",
    ]
)

# TAB 1: Model Comparison
with tabs[0]:
    st.subheader("Model Performance Comparison")

    if df_summary is not None and not df_summary.empty:
        # Pivot for comparison if we have both models
        st.dataframe(
            df_summary.style.format(
                {
                    "auc": "{:.4f}",
                    "ks": "{:.4f}",
                    "gini": "{:.4f}",
                    "accuracy": "{:.4f}",
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1": "{:.4f}",
                }
            ).background_gradient(subset=["auc", "ks", "gini"], cmap="Greens"),
            use_container_width=True,
        )

        # Metrics comparison chart
        if len(df_summary) >= 2:
            metrics_to_compare = ["auc", "ks", "gini"]
            metrics_available = [m for m in metrics_to_compare if m in df_summary.columns]

            if metrics_available:
                fig = go.Figure()
                for _, row in df_summary.iterrows():
                    fig.add_trace(
                        go.Bar(
                            name=row.get("model", "Model"),
                            x=metrics_available,
                            y=[row.get(m, 0) for m in metrics_available],
                        )
                    )

                fig.update_layout(
                    title="Model Performance Comparison",
                    barmode="group",
                    yaxis_title="Score",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Key insights
        st.subheader("Key Insights")
        col1, col2, col3 = st.columns(3)

        if "auc" in df_summary.columns:
            idx = df_summary["auc"].idxmax()
            best_model = df_summary.iloc[idx]
            col1.metric("Best Model", str(best_model.get("model", "N/A")))
            col2.metric("Best AUC", f"{best_model.get('auc', 0):.4f}")
            col3.metric("Best KS", f"{best_model.get('ks', 0):.4f}")
        else:
            col1.metric("Best Model", "N/A")
            col2.metric("Best AUC", "N/A")
            col3.metric("Best KS", "N/A")

    else:
        st.warning("Model summary not available. Run the DS pipeline first.")

        # Show expected output structure
        st.info("""
        Expected metrics from model training:
        - **AUC**: Area Under ROC Curve (0.5 = random, 1.0 = perfect)
        - **KS**: Kolmogorov-Smirnov Statistic (measures separation)
        - **Gini**: 2*AUC - 1 (industry standard metric)
        - **Accuracy, Precision, Recall, F1**: Classification metrics
        """)

# TAB 2: ROC & KS Analysis
with tabs[1]:
    st.subheader("ROC Curve & KS Analysis")

    # --- Model selection ---
    summary_key = paths["outputs"] + "ds3_lr_summary.csv"
    df_summary = load_csv_from_minio(summary_key)
    model_options = (
        [row["model"] for _, row in df_summary.iterrows()]
        if df_summary is not None
        else ["Default"]
    )
    selected_model = st.selectbox(
        "Select model for ROC & KS analysis",
        model_options,
        index=0,
        help="Choose which model's ROC, KS, Gini, and learning curve to display.",
    )

    # Load both train and test data for split ROC/KS
    test_woe_key = paths["processed"] + "test_woe.csv"
    train_woe_key = paths["processed"] + "train_woe.csv"
    df_test = load_csv_from_minio(test_woe_key)
    df_train = load_csv_from_minio(train_woe_key)

    if (
        df_test is not None
        and df_train is not None
        and "default_90p_12m" in df_test.columns
        and "default_90p_12m" in df_train.columns
    ):
        # Prepare test
        y_true_test = df_test["default_90p_12m"].to_numpy()
        X_test = df_test.drop(columns=["default_90p_12m"]).fillna(0).to_numpy()
        y_true_train = df_train["default_90p_12m"].to_numpy()
        X_train = df_train.drop(columns=["default_90p_12m"]).fillna(0).to_numpy()

        # --- Load actual model predictions from MinIO ---
        pred_key = paths["outputs"] + f"ds3_lr_predictions_{selected_model}.csv"
        df_pred = load_csv_from_minio(pred_key)
        # Try to load train predictions if available
        train_pred_key = paths["outputs"] + f"ds3_lr_predictions_{selected_model}_train.csv"
        df_pred_train = load_csv_from_minio(train_pred_key)
        if df_pred is not None and not df_pred.empty:
            y_true_test = df_pred["y_true"].to_numpy()
            y_prob_test = df_pred["y_prob"].to_numpy()
            y_pred_test = df_pred["y_pred"].to_numpy()
            # Use real train predictions if available
            if df_pred_train is not None and not df_pred_train.empty:
                y_true_train = df_pred_train["y_true"].to_numpy()
                y_prob_train = df_pred_train["y_prob"].to_numpy()
            else:
                # Train predictions not available — use test-only mode
                y_true_train = y_true_test
                y_prob_train = y_prob_test
        else:
            # MRM-F13: No synthetic fallback — require real predictions
            st.error(
                f"⚠️ Predictions file not found for model '{selected_model}'. "
                "Please run the DS pipeline to generate predictions. "
                "Synthetic curves have been removed to ensure statistical integrity."
            )
            st.stop()

        from sklearn.metrics import auc, roc_curve

        # ROC for train
        fpr_train, tpr_train, _ = roc_curve(y_true_train, y_prob_train)
        roc_auc_train = auc(fpr_train, tpr_train)
        gini_train = 2 * roc_auc_train - 1
        # ROC for test
        fpr_test, tpr_test, _ = roc_curve(y_true_test, y_prob_test)
        roc_auc_test = auc(fpr_test, tpr_test)
        gini_test = 2 * roc_auc_test - 1

        col1, col2 = st.columns(2)
        with col1:
            # Split ROC
            fig = go.Figure()
            # Adaptive legend and curve display
            if fpr_train is not None and tpr_train is not None:
                fig.add_trace(
                    go.Scatter(
                        x=fpr_train,
                        y=tpr_train,
                        name=f"Train ROC (AUC={roc_auc_train:.4f}, Gini={gini_train:.4f})",
                        line=dict(color="#1abc9c", width=2, dash="solid"),
                    )
                )
            if fpr_test is not None and tpr_test is not None:
                fig.add_trace(
                    go.Scatter(
                        x=fpr_test,
                        y=tpr_test,
                        name=f"Test ROC (AUC={roc_auc_test:.4f}, Gini={gini_test:.4f})",
                        line=dict(color="#3498db", width=2, dash="dash"),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    name="Random",
                    line=dict(color="gray", dash="dot"),
                )
            )
            fig.update_layout(
                title=f"Train vs Test ROC Curve ({selected_model})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Learning Curve (AUC) — loaded from precomputed CSV (MRM-F8)
            lc_key = cfg["minio"]["paths"]["outputs"] + "ds3_learning_curve.csv"
            lc_df = load_csv_from_minio(lc_key)

            if lc_df is not None and not lc_df.empty:
                train_sizes = lc_df["train_size"].to_numpy()
                train_scores_mean = lc_df["train_auc_mean"].to_numpy()
                val_scores_mean = lc_df["val_auc_mean"].to_numpy()

                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scatter(
                        x=train_sizes,
                        y=train_scores_mean,
                        name="Train AUC",
                        line=dict(color="#1abc9c"),
                    )
                )
                fig2.add_trace(
                    go.Scatter(
                        x=train_sizes,
                        y=val_scores_mean,
                        name="Validation AUC",
                        line=dict(color="#e67e22"),
                    )
                )
                fig2.update_layout(
                    title=f"Learning Curve (AUC vs. Training Size) ({selected_model})",
                    xaxis_title="Training Samples",
                    yaxis_title="AUC",
                    xaxis=dict(tickformat=",d"),
                    yaxis=dict(range=[0.5, 1.0]),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Learning curve data not available. Run the DS pipeline to generate.")

        with col2:
            # KS Curve (Test)
            sorted_indices = np.argsort(y_prob_test)
            y_true_sorted = y_true_test[sorted_indices]
            total_positives = np.sum(y_true_test)
            total_negatives = len(y_true_test) - total_positives
            cumulative_positives = np.cumsum(y_true_sorted) / total_positives
            cumulative_negatives = np.cumsum(1 - y_true_sorted) / total_negatives
            ks_stat = np.max(np.abs(cumulative_positives - cumulative_negatives))
            ks_idx = np.argmax(np.abs(cumulative_positives - cumulative_negatives))
            x_axis = np.arange(len(y_true_test)) / len(y_true_test) * 100
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=cumulative_positives * 100,
                    name="Bad (Defaults)",
                    line=dict(color="#e74c3c"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=cumulative_negatives * 100,
                    name="Good",
                    line=dict(color="#2ecc71"),
                )
            )
            fig.add_annotation(
                x=x_axis[ks_idx],
                y=(cumulative_positives[ks_idx] + cumulative_negatives[ks_idx]) / 2 * 100,
                text=f"KS = {ks_stat:.4f}",
                showarrow=True,
            )
            fig.add_vline(
                x=x_axis[ks_idx],
                line_dash="dash",
                line_color="#8e44ad",
                annotation_text="KS-max",
                annotation_position="top right",
            )
            fig.update_layout(
                title=f"KS Curve (Test) ({selected_model})",
                xaxis_title="Population %",
                yaxis_title="Cumulative %",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Summary metrics for selected model (always from df_summary)
        col1, col2, col3, col4, col5 = st.columns(5)
        if df_summary is not None and selected_model in df_summary["model"].to_numpy():
            model_row = df_summary[df_summary["model"] == selected_model].iloc[0]
            col1.metric("AUC", f"{model_row.get('auc', 0):.4f}")
            col2.metric("KS", f"{model_row.get('ks', 0):.4f}")
            col3.metric("Gini", f"{model_row.get('gini', 0):.4f}")
            col4.metric("F1", f"{model_row.get('f1', 0):.4f}")
            col5.metric("Recall", f"{model_row.get('recall', 0):.4f}")
        else:
            col1.metric("AUC", "N/A")
            col2.metric("KS", "N/A")
            col3.metric("Gini", "N/A")
            col4.metric("F1", "N/A")
            col5.metric("Recall", "N/A")
    else:
        st.warning("Train or test data not available for curve generation.")

# TAB 3: Feature Importance
with tabs[2]:
    st.subheader("Feature Importance Analysis")

    if df_importance is not None and not df_importance.empty:
        # Top N selector
        n_features = st.slider("Number of features to display", 5, 50, 20)

        df_top = df_importance.head(n_features)

        # Bar chart
        fig = px.bar(
            df_top,
            x=("abs_coefficient" if "abs_coefficient" in df_top.columns else "coefficient"),
            y="feature",
            orientation="h",
            title=f"Top {n_features} Features by Importance",
            color="coefficient" if "coefficient" in df_top.columns else None,
            color_continuous_scale="RdBu_r",
        )
        fig.update_layout(height=max(400, n_features * 25))
        st.plotly_chart(fig, use_container_width=True)

        # Risk direction breakdown
        if "risk_direction" in df_importance.columns:
            col1, col2 = st.columns(2)

            with col1:
                direction_counts = df_importance["risk_direction"].value_counts()
                fig = px.pie(
                    values=direction_counts.to_numpy(),
                    names=direction_counts.index,
                    title="Risk Direction Distribution",
                    color_discrete_sequence=["#e74c3c", "#2ecc71"],
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "relative_importance" in df_importance.columns:
                    st.subheader("Feature Concentration")

                    # How many features for 80% importance
                    cumsum = df_importance["relative_importance"].cumsum()
                    n_80 = (cumsum <= 80).sum() + 1

                    st.metric(
                        "Features for 80% importance",
                        n_80,
                        delta=f"{n_80 / len(df_importance) * 100:.1f}% of all features",
                    )

                    # Cumulative importance chart
                    fig = px.line(
                        x=range(1, len(cumsum) + 1),
                        y=cumsum.to_numpy(),
                        title="Cumulative Importance",
                        labels={
                            "x": "Number of Features",
                            "y": "Cumulative Importance %",
                        },
                    )
                    fig.add_hline(y=80, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)

        # Full table
        with st.expander("Full Feature Importance Table"):
            st.dataframe(df_importance, use_container_width=True)

            # Download button
            csv_buffer = df_importance.to_csv(index=False)
            st.download_button(
                label="Download Feature Importance CSV",
                data=csv_buffer,
                file_name="feature_importance.csv",
                mime="text/csv",
            )
    else:
        st.warning("Feature importance not available. Run the DS pipeline first.")

# TAB 4: IV Analysis
with tabs[3]:
    st.subheader("Information Value (IV) Analysis")

    if df_iv is not None and not df_iv.empty:
        # IV interpretation guide
        st.info("""
        **IV Interpretation:**
        - IV < 0.02: Not predictive
        - 0.02 ≤ IV < 0.1: Weak
        - 0.1 ≤ IV < 0.3: Medium
        - 0.3 ≤ IV < 0.5: Strong
        - IV ≥ 0.5: Suspiciously strong
        """)

        # IV distribution
        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                df_iv.head(30),
                x="iv" if "iv" in df_iv.columns else df_iv.columns[1],
                y="feature" if "feature" in df_iv.columns else df_iv.columns[0],
                orientation="h",
                title="Top 30 Features by IV",
                color="iv" if "iv" in df_iv.columns else None,
                color_continuous_scale="Viridis",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # IV category breakdown
            iv_col = "iv" if "iv" in df_iv.columns else df_iv.columns[1]

            categories = pd.cut(
                df_iv[iv_col],
                bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
                labels=["Not Predictive", "Weak", "Medium", "Strong", "Suspicious"],
            )
            cat_counts = categories.value_counts()

            fig = px.pie(
                values=cat_counts.to_numpy(),
                names=cat_counts.index,
                title="IV Distribution",
                color_discrete_map={
                    "Not Predictive": "#95a5a6",
                    "Weak": "#f39c12",
                    "Medium": "#27ae60",
                    "Strong": "#2980b9",
                    "Suspicious": "#e74c3c",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            st.subheader("IV Summary")
            st.write(f"**Total features:** {len(df_iv)}")
            st.write(f"**Predictive (IV≥0.02):** {(df_iv[iv_col] >= 0.02).sum()}")
            st.write(f"**Strong (IV≥0.3):** {(df_iv[iv_col] >= 0.3).sum()}")
            st.write(f"**Mean IV:** {df_iv[iv_col].mean():.4f}")
            st.write(f"**Max IV:** {df_iv[iv_col].max():.4f}")

        # Flagged features
        suspicious = df_iv[df_iv[iv_col] >= 0.5]
        if len(suspicious) > 0:
            st.warning(f"⚠️ {len(suspicious)} features have IV ≥ 0.5 - Check the data")
            st.dataframe(suspicious, use_container_width=True)
    else:
        st.warning("IV analysis not available. Run feature engineering first.")

# TAB 5: MLflow Experiments
with tabs[4]:
    st.subheader("MLflow Experiment Tracking")

    df_runs = get_mlflow_experiments()

    if not df_runs.empty:
        # Filter columns of interest
        metric_cols = [c for c in df_runs.columns if c.startswith("metrics.")]
        param_cols = [c for c in df_runs.columns if c.startswith("params.")]

        display_cols = [
            "run_id",
            "experiment_name",
            "status",
            "start_time",
        ] + metric_cols[:10]
        display_cols = [c for c in display_cols if c in df_runs.columns]

        st.dataframe(
            df_runs[display_cols].sort_values("start_time", ascending=False),
            use_container_width=True,
        )

        # Metrics comparison across runs
        if metric_cols:
            selected_metric = st.selectbox("Select metric to compare", metric_cols)

            if selected_metric in df_runs.columns:
                fig = px.bar(
                    df_runs.dropna(subset=[selected_metric]).head(20),
                    x="run_id",
                    y=selected_metric,
                    color=("experiment_name" if "experiment_name" in df_runs.columns else None),
                    title=f"{selected_metric} Across Runs",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No MLflow experiments found. Start training models to see results here.")
        st.markdown(f"""
        **MLflow Tracking URI:** `{cfg["mlflow"]["tracking_uri"]}`

        Make sure MLflow server is running and models have been trained.
        """)
