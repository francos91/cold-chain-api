import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Risk Intelligence", layout="wide")
st.title("Predictive Risk Intelligence")

API_URL = "https://cold-chain-api-docker.onrender.com/api/dashboard"

@st.cache_data(ttl=600)
def get_risk_data():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data["features"])
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        return pd.DataFrame()

with st.spinner("Fetching pre-calculated ML data..."):
    df_ready = get_risk_data()

if df_ready.empty:
    st.warning("No data available.")
    st.stop()

st.info(f"Total Shipments Evaluated: `{len(df_ready):,}`")

# Risk Assessment UI
threshold_pct = st.slider("Risk Threshold (%)", 5, 95, 20, 5, format="%d%%")
threshold = threshold_pct / 100.0

df_ready["predicted_risk"] = (df_ready["risk_probability"] >= threshold).astype(int)
high_risk_count = int(df_ready["predicted_risk"].sum())
total_count = len(df_ready)
high_risk_share = (high_risk_count / total_count) * 100 if total_count > 0 else 0
max_model_prob = (df_ready["risk_probability"].max() * 100) if len(df_ready) > 0 else 0

col1, col2 = st.columns(2)
col1.metric("Predicted High-Risk Shipments", high_risk_count)
col2.metric("High-Risk Fleet Share", f"{high_risk_share:.1f}%")

st.caption(f"Model max probability score in current view: **{max_model_prob:.1f}%**.")
st.divider()

if high_risk_count > 0:
    high_risk_df = df_ready[(df_ready["predicted_risk"] == 1) & (df_ready["status"] != "delivered")]
    top_5_df = high_risk_df.sort_values("risk_probability", ascending=False).head(5)
    
    st.markdown("### Top 5 Highest Risk Shipments")
    st.dataframe(
        top_5_df[["shipment_id", "origin", "destination", "temperature_celsius", "status", "risk_probability"]],
        width="stretch",
        column_config={
            "risk_probability": st.column_config.NumberColumn("Risk Probability", format="%.1f%%"),
            "temperature_celsius": st.column_config.NumberColumn("Temp (°C)", format="%.2f")
        }
    )
else:
    st.success(f"No active high-risk shipments predicted at the **{threshold_pct}%** threshold!")

st.divider()

st.markdown("### High-Risk Routes Analysis")
route_risk = df_ready.groupby("route")["is_risk"].mean().reset_index()
fig_risk = px.bar(
    route_risk.sort_values("is_risk", ascending=False).head(10),
    x="is_risk",
    y="route",
    orientation="h",
    title="Routes with Highest Excursion Frequency",
)
st.plotly_chart(fig_risk, width="stretch")
