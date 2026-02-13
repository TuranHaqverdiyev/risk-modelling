import streamlit as st


st.set_page_config(
    page_title="Credit Risk Modelling Hub",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Credit Risk Modelling Platform")

st.markdown("""
Welcome to the **End-to-End Platform for credit risk modelling**.
This platform provides tools for data analysis, model performance, and monitoring.

### 🚀 Quick Access
Choose a module from the sidebar:

*   **📊 Portfolio Analytics**: Dive into the dataset and portfolio characteristics.
*   **🎮 Cutoff Simulator**: Simulate the impact of different credit score cutoffs.
*   **📈 Model Performance**: Evaluate the performance of trained models.
*   **🖥️ Monitoring**: Monitor data and model drift in production.

""")
