import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Thermal Tracking", layout="wide")
st.title("Thermal Profile & GPS Route Analysis")
st.markdown("View the temperature curve and GPS route for a shipment, powered by the Render Telemetry API.")

API_DASHBOARD = "https://cold-chain-api-docker.onrender.com/api/dashboard"
API_TELEMETRY = "https://cold-chain-api-docker.onrender.com/api/telemetry"

@st.cache_data(ttl=600)
def get_base_shipments():
    try:
        res = requests.get(API_DASHBOARD)
        res.raise_for_status()
        return pd.DataFrame(res.json()["raw"])
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_telemetry_stream():
    try:
        res = requests.get(API_TELEMETRY)
        res.raise_for_status()
        return pd.DataFrame(res.json())
    except Exception as e:
        st.error(f"Failed to fetch telemetry API: {e}")
        return pd.DataFrame()

df = get_base_shipments()

if df.empty:
    st.warning("No shipment data available from API.")
    st.stop()

# Sidebar for selection
st.sidebar.header("Filter Shipments")
selected_status = st.sidebar.selectbox("Select Status", options=df["status"].unique())

filtered_df = df[df["status"] == selected_status]

if "thermal_loaded" not in st.session_state:
    st.session_state.thermal_loaded = False

if st.button("Load Thermal Telemetry & GPS Map") or st.session_state.thermal_loaded:
    st.session_state.thermal_loaded = True

    with st.spinner("Fetching telemetry stream from Render API..."):
        ts_df = get_telemetry_stream()

    if ts_df.empty:
        st.warning("No time-series data found.")
    else:
        # Filter available shipments by status
        available_ids = filtered_df["shipment_id"].unique()
        ts_df_filtered = ts_df[ts_df["shipment_id"].isin(available_ids)]

        if ts_df_filtered.empty:
            st.info(f"No telemetry data found for status: {selected_status}")
        else:
            shipment_ids = ts_df_filtered["shipment_id"].unique()
            selected_id = st.selectbox("Select Shipment to Analyze", shipment_ids)

            ts_selected = ts_df_filtered[ts_df_filtered["shipment_id"] == selected_id].sort_values("time_minutes")
            
            st.subheader("Quick Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Shipment", selected_id)
            col2.metric("Status", selected_status.capitalize())
            col3.metric("Avg Temp", f"{ts_selected['temperature'].mean():.1f}°C")

            st.subheader("Comparison to Average")
            same_status_data = ts_df_filtered
            
            if len(same_status_data) > 0:
                avg_per_shipment = same_status_data.groupby("shipment_id")["temperature"].mean()
                status_avg_temp = avg_per_shipment.mean()
                status_std_temp = avg_per_shipment.std()
                selected_avg_temp = ts_selected["temperature"].mean()
                temp_diff = selected_avg_temp - status_avg_temp

                col1, col2, col3 = st.columns(3)
                col1.metric(f"Status Average ({selected_status})", f"{status_avg_temp:.1f}°C")
                col2.metric("Your Shipment", f"{selected_avg_temp:.1f}°C")

                if temp_diff > 0:
                    col3.metric("Difference", f"+{temp_diff:.1f}°C", delta="Warmer than average", delta_color="inverse")
                else:
                    col3.metric("Difference", f"{temp_diff:.1f}°C", delta="Cooler than average", delta_color="normal")

            # Chart
            st.subheader("Temperature Profile")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_selected["time_minutes"], y=ts_selected["temperature"],
                mode="lines+markers", name=f"{selected_id}", line=dict(color="blue", width=3)
            ))

            if len(same_status_data) > 0:
                time_points = ts_selected["time_minutes"].values
                all_curves = []
                for sid in avg_per_shipment.index[:10]:
                    curve = ts_df_filtered[ts_df_filtered["shipment_id"] == sid].sort_values("time_minutes")["temperature"].values
                    if len(curve) == len(time_points):
                        all_curves.append(curve)

                if len(all_curves) > 0:
                    avg_curve = np.mean(all_curves, axis=0)
                    std_curve = np.std(all_curves, axis=0)
                    fig.add_trace(go.Scatter(
                        x=time_points, y=avg_curve, mode="lines",
                        name=f"Average", line=dict(color="orange", width=2, dash="dash")
                    ))

            fig.add_hline(y=8.0, line_dash="dash", line_color="green", annotation_text="Breach Threshold (8°C)")
            fig.update_layout(height=450, hovermode="x unified", xaxis_title="Time (min)", yaxis_title="Temp (°C)")
            st.plotly_chart(fig, width="stretch")

            # Map
            st.subheader("GPS Route (Colored by Temperature)")
            center_lat, center_lon = ts_selected["latitude"].mean(), ts_selected["longitude"].mean()
            
            fig_map = px.scatter_map(
                ts_selected, lat="latitude", lon="longitude", color="temperature",
                hover_name="time_minutes", color_continuous_scale="RdYlBu_r", zoom=6, height=450
            )
            fig_map.update_layout(map_style="open-street-map", map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=6))
            st.plotly_chart(fig_map, width="stretch")
