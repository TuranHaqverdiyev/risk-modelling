"""
Portfolio Analytics Dashboard
"""

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.io import get_s3_client as _get_s3, load_config as _load_cfg


st.set_page_config(page_title="Portfolio Analytics", layout="wide")

st.header("📊 Portfolio Analytics")


# Load config
@st.cache_resource
def load_config():
    return _load_cfg()


@st.cache_resource
def get_s3_client():
    cfg = load_config()
    return _get_s3(cfg)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_csv_from_minio(key: str) -> pd.DataFrame:
    """Load CSV from MinIO with caching."""
    try:
        cfg = load_config()
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=cfg["minio"]["bucket"], Key=key)
        return pd.read_csv(BytesIO(obj["Body"].read()))
    except Exception as e:
        st.error(f"Could not load {key}: {e}")
        return pd.DataFrame()  # Return empty DataFrame for error cases


@st.cache_data(ttl=3600)  # Cache for 1 hour (xlsx is slow to parse)
def load_xlsx_from_minio(key: str) -> pd.DataFrame:
    """Load Excel file from MinIO with caching.
    Note: Excel parsing is slow (~7s), but cached for 1 hour.
    """
    try:
        cfg = load_config()
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=cfg["minio"]["bucket"], Key=key)
        return pd.read_excel(BytesIO(obj["Body"].read()), engine="openpyxl")
    except Exception as e:
        st.error(f"Could not load {key}: {e}")
        return None  # type: ignore


@st.cache_data(ttl=3600)
def load_full_dataset_fast() -> pd.DataFrame:
    """Load full dataset - prefer CSV cache over slow xlsx."""
    cfg = load_config()
    s3 = get_s3_client()
    bucket = cfg["minio"]["bucket"]

    # Try to load pre-converted CSV first (much faster)
    csv_key = "processed/full_dataset.csv"

    try:
        obj = s3.get_object(Bucket=bucket, Key=csv_key)
        return pd.read_csv(BytesIO(obj["Body"].read()))
    except Exception as e:
        st.error(f"Could not load {csv_key}: {e}")

    # Fall back to xlsx and save as CSV for next time
    xlsx_key = cfg["minio"]["paths"]["raw"] + "dataset.xlsx"
    obj = s3.get_object(Bucket=bucket, Key=xlsx_key)
    df = pd.read_excel(BytesIO(obj["Body"].read()), engine="openpyxl")

    # Save as CSV for faster future loads
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=csv_key, Body=buffer.getvalue())

    return df


# Sidebar: Data source selection
st.sidebar.header("Data Settings")
data_source = st.sidebar.radio("Data Source", ["Training Set", "Test Set", "Full Dataset"])

cfg = load_config()
paths = cfg["minio"]["paths"]

# Load appropriate data based on selection
with st.spinner(f"Loading {data_source}..."):
    if data_source == "Full Dataset":
        df = load_full_dataset_fast()  # Uses CSV cache for speed
    else:
        data_key_map = {
            "Training Set": paths["processed"] + "train.csv",
            "Test Set": paths["processed"] + "test.csv",
        }
        df = load_csv_from_minio(data_key_map[data_source])

if df is not None:
    st.success(f"Loaded {len(df):,} records from {data_source}")

    # Tabs for different analytics sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📋 Dataset Overview",
            "💰 Income & Employment",
            "📅 DPD Behavior",
            "🔍 Inquiry Analysis",
            "💳 Affordability",
        ]
    )

    # TAB 1: Dataset Overview
    with tab1:
        st.subheader("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        target_col = "default_90p_12m"
        if target_col in df.columns:
            default_rate = df[target_col].mean() * 100
            col1.metric("Records", f"{len(df):,}")
            col2.metric("Features", f"{len(df.columns)}")
            col3.metric("Default Rate", f"{default_rate:.2f}%")
            col4.metric("Good Rate", f"{100 - default_rate:.2f}%")

            # Target distribution - use explicit value counts for correct mapping
            target_counts = df[target_col].value_counts().sort_index()
            pie_df = pd.DataFrame(
                {
                    "Status": ["Good (0)", "Bad (1)"],
                    "Count": [target_counts.get(0, 0), target_counts.get(1, 0)],
                }
            )

            fig = px.pie(
                pie_df,
                values="Count",
                names="Status",
                title="Target Distribution",
                color="Status",
                color_discrete_map={"Good (0)": "#2ecc71", "Bad (1)": "#e74c3c"},
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        # Missing values analysis
        st.subheader("Missing Values Analysis")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = (
            pd.DataFrame(
                {
                    "Feature": missing.index,
                    "Missing Count": missing.to_numpy(),
                    "Missing %": missing_pct.to_numpy(),
                }
            )
            .query("`Missing Count` > 0")
            .sort_values("Missing %", ascending=False)
        )

        if len(missing_df) > 0:
            fig = px.bar(
                missing_df.head(20),
                x="Feature",
                y="Missing %",
                title="Missing Rate",
                color="Missing %",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values in the dataset")

        # Data types
        with st.expander("Column Information"):
            dtypes_df = pd.DataFrame(
                {
                    "Column": df.columns.tolist(),
                    "Type": df.dtypes.astype(str).tolist(),
                    "Non-Null": df.count().to_numpy().tolist(),
                    "Unique": df.nunique().to_numpy().tolist(),
                }
            )
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    # TAB 2: Income & Employment
    with tab2:
        st.subheader("Income & Employment Analysis")

        # Find income-related columns
        income_cols = [
            c
            for c in df.columns
            if any(x in c.lower() for x in ["income", "salary", "employment", "employer"])
        ]

        if income_cols:
            selected_col = st.selectbox("Select Income Feature", income_cols)

            # Check if numeric or categorical
            is_numeric = df[selected_col].dtype in [
                "float64",
                "int64",
                "float32",
                "int32",
            ]

            if is_numeric:
                # Distribution Plot
                fig = px.histogram(
                    df,
                    x=selected_col,
                    nbins=50,
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=["#3498db"],
                )
                st.plotly_chart(fig, use_container_width=True)

                # Box Plot
                if target_col in df.columns:
                    fig = px.box(
                        df,
                        x=target_col,
                        y=selected_col,
                        title=f"{selected_col} by Default Status",
                        color=target_col,
                        color_discrete_sequence=["#2ecc71", "#e74c3c"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Summary statistics
                st.subheader("Summary Statistics")
                if target_col in df.columns:
                    summary = (
                        df.groupby(target_col)[selected_col]
                        .agg(["count", "mean", "median", "std", "min", "max"])
                        .round(2)
                    )
                    summary.index = ["Good", "Bad"]
                    st.dataframe(summary, use_container_width=True)
            else:
                # Categorical feature handling
                # Value counts bar chart
                value_counts = df[selected_col].value_counts().head(20).reset_index()
                value_counts.columns = [selected_col, "Count"]
                fig = px.bar(
                    value_counts,
                    x=selected_col,
                    y="Count",
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=["#3498db"],
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

                if target_col in df.columns:
                    # Default rate by category
                    default_by_cat = (
                        df.groupby(selected_col)[target_col].agg(["mean", "count"]).reset_index()
                    )
                    default_by_cat.columns = [selected_col, "Default Rate", "Count"]
                    default_by_cat["Default Rate"] = (default_by_cat["Default Rate"] * 100).round(2)
                    default_by_cat = default_by_cat.sort_values(
                        "Default Rate", ascending=False
                    ).head(20)

                    fig = px.bar(
                        default_by_cat,
                        x=selected_col,
                        y="Default Rate",
                        title=f"Default Rate by {selected_col}",
                        color="Default Rate",
                        color_continuous_scale="RdYlGn_r",
                        text="Count",
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

                # Summary table for categorical
                st.subheader("Category Summary")
                if target_col in df.columns:
                    cat_summary = (
                        df.groupby(selected_col)
                        .agg({target_col: ["count", "sum", "mean"]})
                        .round(4)
                    )
                    cat_summary.columns = ["Total", "Defaults", "Default Rate"]
                    cat_summary["Default Rate"] = (cat_summary["Default Rate"] * 100).round(2)
                    cat_summary["% of Total"] = (
                        cat_summary["Total"] / cat_summary["Total"].sum() * 100
                    ).round(2)
                    cat_summary = cat_summary.sort_values("Total", ascending=False)
                    st.dataframe(cat_summary, use_container_width=True)
        else:
            st.info("No income-related columns found in the dataset")

    # TAB 3: DPD Behavior
    with tab3:
        st.subheader("Days Past Due (DPD) Analysis")

        # Find DPD-related columns
        dpd_cols = [
            c
            for c in df.columns
            if any(x in c.lower() for x in ["dpd", "delinq", "past_due", "overdue"])
        ]

        if dpd_cols:
            st.write(f"Found {len(dpd_cols)} DPD-related features")

            selected_dpd = st.multiselect(
                "Select DPD Features to Analyze",
                dpd_cols,
                default=dpd_cols[:3] if len(dpd_cols) >= 3 else dpd_cols,
            )

            if selected_dpd:
                # Correlation with default
                if target_col in df.columns:
                    correlations = []
                    for col in selected_dpd:
                        if df[col].dtype in ["float64", "int64"]:
                            corr = df[col].corr(df[target_col])
                            correlations.append({"Feature": col, "Correlation with Default": corr})

                    if correlations:
                        corr_df = pd.DataFrame(correlations).sort_values(
                            "Correlation with Default", ascending=False
                        )
                        fig = px.bar(
                            corr_df,
                            x="Correlation with Default",
                            y="Feature",
                            orientation="h",
                            title="DPD Features Correlation with Default",
                            color="Correlation with Default",
                            color_continuous_scale="RdBu_r",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Distribution of selected DPD features
                for col in selected_dpd[:3]:
                    if df[col].dtype in ["float64", "int64"]:
                        fig = px.histogram(
                            df,
                            x=col,
                            color=target_col if target_col in df.columns else None,
                            barmode="overlay",
                            title=f"Distribution of {col}",
                            opacity=0.7,
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No DPD-related columns found. Common names: dpd_, delinq_, past_due_")

    # TAB 4: Inquiry Analysis
    with tab4:
        st.subheader("Credit Inquiry Analysis")

        # Find inquiry-related columns
        inq_cols = [
            c for c in df.columns if any(x in c.lower() for x in ["inq", "inquiry", "enquir"])
        ]

        if inq_cols:
            selected_inq = st.selectbox("Select Inquiry Feature", inq_cols)

            # Value counts for inquiry features
            if df[selected_inq].nunique() <= 20:
                value_counts = df[selected_inq].value_counts().reset_index()
                value_counts.columns = [selected_inq, "Count"]
                fig = px.bar(
                    value_counts,
                    x=selected_inq,
                    y="Count",
                    title=f"Distribution of {selected_inq}",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(
                    df,
                    x=selected_inq,
                    nbins=30,
                    title=f"Distribution of {selected_inq}",
                )
                st.plotly_chart(fig, use_container_width=True)

            if target_col in df.columns:
                # Default rate by inquiry bucket
                if df[selected_inq].dtype in ["float64", "int64"]:
                    df_temp = df.copy()
                    df_temp[f"{selected_inq}_bucket"] = pd.cut(
                        df_temp[selected_inq], bins=5, duplicates="drop"
                    )
                    default_by_bucket = df_temp.groupby(f"{selected_inq}_bucket")[target_col].agg(
                        ["mean", "count"]
                    )
                    default_by_bucket.columns = ["Default Rate", "Count"]
                    default_by_bucket = default_by_bucket.reset_index()
                    default_by_bucket[f"{selected_inq}_bucket"] = default_by_bucket[
                        f"{selected_inq}_bucket"
                    ].astype(str)

                    fig = px.bar(
                        default_by_bucket,
                        x=f"{selected_inq}_bucket",
                        y="Default Rate",
                        title=f"Default Rate by {selected_inq}",
                        text="Count",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inquiry-related columns found")

    # TAB 5: Affordability
    with tab5:
        st.subheader("Affordability Analysis")

        # Find affordability-related columns
        afford_cols = [
            c
            for c in df.columns
            if any(
                x in c.lower()
                for x in [
                    "dti",
                    "pti",
                    "ltv",
                    "debt",
                    "loan_amount",
                    "payment",
                    "ratio",
                ]
            )
        ]

        if afford_cols:
            st.write(f"Found {len(afford_cols)} affordability-related features")

            # Feature selection for scatter plot
            x_feat = st.selectbox("X-axis Feature", afford_cols, index=0)
            y_feat = st.selectbox("Y-axis Feature", afford_cols, index=min(1, len(afford_cols) - 1))

            # Scatter plot
            if target_col in df.columns:
                fig = px.scatter(
                    df.sample(min(5000, len(df))),  # Sample for performance
                    x=x_feat,
                    y=y_feat,
                    color=target_col,
                    title=f"{y_feat} vs {x_feat}",
                    opacity=0.5,
                    color_discrete_sequence=["#2ecc71", "#e74c3c"],
                )
                st.plotly_chart(fig, use_container_width=True)

            # Summary table
            if target_col in df.columns:
                st.subheader("Affordability Metrics by Default Status")
                # Filter to only numeric columns
                numeric_afford_cols = [
                    c
                    for c in afford_cols
                    if df[c].dtype in ["float64", "int64", "float32", "int32"]
                ]
                if numeric_afford_cols:
                    summary = df.groupby(target_col)[numeric_afford_cols].mean().T.round(4)
                    summary.columns = ["Good", "Bad"]
                    summary["Difference"] = summary["Bad"] - summary["Good"]
                    summary["Diff %"] = ((summary["Bad"] / summary["Good"] - 1) * 100).round(2)
                    st.dataframe(
                        summary.sort_values("Diff %", ascending=False),
                        use_container_width=True,
                    )
                else:
                    st.info("No numeric affordability columns found")
        else:
            st.info("No affordability-related columns found")

else:
    st.error("Could not load data. Please check MinIO connection and data availability.")
