"""
Monitoring Dashboard

Displays model monitoring outputs:
- PSI / CSI (Population Stability Index)
- Performance Alerts (AUC, KS, Gini degradation)
- Loss Alerts (actual vs expected default rates)
"""

from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.io import get_s3_client as _get_s3, load_config as _load_cfg


st.set_page_config(page_title="Monitoring", layout="wide")

st.header("📡 Model Monitoring")


# ── Config & I/O ───────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    return _load_cfg()


@st.cache_resource
def get_s3_client():
    cfg = load_config()
    return _get_s3(cfg)


@st.cache_data(ttl=3600)
def load_csv_from_minio(key: str) -> pd.DataFrame:
    """Load CSV from MinIO with caching."""
    try:
        cfg = load_config()
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=cfg["minio"]["bucket"], Key=key)
        return pd.read_csv(BytesIO(obj["Body"].read()))
    except Exception:
        return pd.DataFrame()


cfg = load_config()
PATHS = cfg["minio"]["paths"]

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 PSI / CSI", "📈 Performance Alerts", "⚠️ Loss Alerts"])

# ===========================================================================
# TAB 1: PSI / CSI
# ===========================================================================
with tab1:
    st.subheader("Population Stability Index (PSI)")

    psi_key = PATHS["monitoring"] + "mon1_psi_report.csv"
    df_psi = load_csv_from_minio(psi_key)

    if df_psi is not None and not df_psi.empty:
        # PSI summary badges
        def psi_badge(val):
            if val < 0.1:
                return "🟢 Stable"
            elif val < 0.25:
                return "🟡 Moderate"
            else:
                return "🔴 Significant"

        if "feature" in df_psi.columns and "psi" in df_psi.columns:
            df_psi["status"] = df_psi["psi"].apply(psi_badge)

            # Summary metrics
            n_stable = (df_psi["psi"] < 0.1).sum()
            n_moderate = ((df_psi["psi"] >= 0.1) & (df_psi["psi"] < 0.25)).sum()
            n_critical = (df_psi["psi"] >= 0.25).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Features", len(df_psi))
            c2.metric("🟢 Stable", int(n_stable))
            c3.metric("🟡 Moderate", int(n_moderate))
            c4.metric("🔴 Significant", int(n_critical))

            # Sorted bar chart
            df_sorted = df_psi.sort_values("psi", ascending=False).head(20)
            colors = df_sorted["psi"].apply(
                lambda x: "#e74c3c" if x >= 0.25 else ("#f39c12" if x >= 0.1 else "#2ecc71")
            )
            fig = go.Figure(
                go.Bar(
                    x=df_sorted["feature"],
                    y=df_sorted["psi"],
                    marker_color=colors.tolist(),
                )
            )
            fig.add_hline(
                y=0.1,
                line_dash="dash",
                line_color="#f39c12",
                annotation_text="Warning (0.1)",
            )
            fig.add_hline(
                y=0.25,
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text="Critical (0.25)",
            )
            fig.update_layout(
                title="PSI by Feature (Top 20)",
                xaxis_title="Feature",
                yaxis_title="PSI",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detail table
            with st.expander("Full PSI Table"):
                st.dataframe(df_psi, use_container_width=True)
        else:
            st.info("PSI data columns not in expected format. Displaying raw data.")
            st.dataframe(df_psi, use_container_width=True)
    else:
        st.info("No PSI data available. Run the monitoring pipeline to generate.")

# ===========================================================================
# TAB 2: Performance Alerts
# ===========================================================================
with tab2:
    st.subheader("Performance Monitoring")

    perf_key = PATHS["monitoring"] + "mon2_performance_report.csv"
    df_perf = load_csv_from_minio(perf_key)

    if df_perf is not None and not df_perf.empty:
        # Show metric gauges
        metric_cols = [c for c in df_perf.columns if c not in ("timestamp", "model", "alert_level")]

        if metric_cols:
            cols = st.columns(min(len(metric_cols), 4))
            for i, col_name in enumerate(metric_cols[:4]):
                val = df_perf[col_name].iloc[-1] if len(df_perf) > 0 else 0
                cols[i].metric(col_name.upper(), f"{val:.4f}")

        # Performance over time (if timestamp column exists)
        if "timestamp" in df_perf.columns and len(df_perf) > 1:
            fig = px.line(
                df_perf,
                x="timestamp",
                y=[c for c in metric_cols if c not in ("tn", "fp", "fn", "tp")],
                title="Metric Trends Over Time",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Alert level distribution
        if "alert_level" in df_perf.columns:
            alert_counts = df_perf["alert_level"].value_counts()
            fig_alert = px.pie(
                values=alert_counts.to_numpy(),
                names=alert_counts.index,
                title="Alert Level Distribution",
                color_discrete_map={
                    "OK": "#2ecc71",
                    "WARNING": "#f39c12",
                    "CRITICAL": "#e74c3c",
                },
            )
            st.plotly_chart(fig_alert, use_container_width=True)

        with st.expander("Full Performance Report"):
            st.dataframe(df_perf, use_container_width=True)
    else:
        st.info("No performance data available. Run the monitoring pipeline to generate.")

# ===========================================================================
# TAB 3: Loss Alerts
# ===========================================================================
with tab3:
    st.subheader("Loss Monitoring & Alerts")

    loss_key = PATHS["monitoring"] + "mon3_loss_alerts.csv"
    df_loss = load_csv_from_minio(loss_key)

    if df_loss is not None and not df_loss.empty:
        # Severity breakdown
        if "severity" in df_loss.columns:
            sev_counts = df_loss["severity"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("ℹ️ Info", int(sev_counts.get("INFO", 0)))
            c2.metric("⚠️ Warning", int(sev_counts.get("WARNING", 0)))
            c3.metric("🚨 Critical", int(sev_counts.get("CRITICAL", 0)))

        # Alert timeline
        if "timestamp" in df_loss.columns:
            df_loss_sorted = df_loss.sort_values("timestamp", ascending=False)

            color_map = {"INFO": "#3498db", "WARNING": "#f39c12", "CRITICAL": "#e74c3c"}
            fig = px.scatter(
                df_loss_sorted,
                x="timestamp",
                y="severity" if "severity" in df_loss.columns else df_loss.columns[0],
                color="severity" if "severity" in df_loss.columns else None,
                color_discrete_map=color_map,
                title="Alert Timeline",
                hover_data=df_loss.columns.tolist(),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Full table
        with st.expander("All Alerts"):
            st.dataframe(df_loss, use_container_width=True)
    else:
        st.info("No loss alerts available. Run the monitoring pipeline to generate.")
