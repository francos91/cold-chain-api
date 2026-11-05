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
                selected_avg_temp = ts_selected["temperature"].mean()
                temp_diff = selected_avg_temp - status_avg_temp

                col1, col2, col3 = st.columns(3)
                col1.metric(f"Status Average ({selected_status})", f"{status_avg_temp:.1f}°C")
                col2.metric("Your Shipment", f"{selected_avg_temp:.1f}°C")

                if temp_diff > 0:
                    col3.metric("Difference", f"+{temp_diff:.1f}°C", delta="Warmer than average", delta_color="inverse")
                    st.warning(f"This shipment is **{temp_diff:.1f}°C warmer** than the average {selected_status} shipment.")
                else:
                    col3.metric("Difference", f"{temp_diff:.1f}°C", delta="Cooler than average", delta_color="normal")
                    st.success(f"This shipment is **{abs(temp_diff):.1f}°C cooler** than the average {selected_status} shipment.")

                breach_count_same = same_status_data[same_status_data["temperature"] > 8.0]["shipment_id"].nunique()
                total_same = len(avg_per_shipment)
                if total_same > 0:
                    breach_rate = (breach_count_same / total_same) * 100
                    st.info(f"**{breach_rate:.1f}%** of {selected_status} shipments had at least one temperature breach (>8°C).")

            # --- TEMPERATURE PROFILE PLOT ---
            st.subheader("Temperature Profile")
            fig = go.Figure()

            # 1. Selected shipment trace
            fig.add_trace(go.Scatter(
                x=ts_selected["time_minutes"],
                y=ts_selected["temperature"],
                mode="lines+markers",
                name=f"{selected_id} ({selected_status})",
                line=dict(color="#1f77b4", width=3)
            ))

            # 2. Average curve & Realistic Safe Operating Tolerance Band (+/- 1.5°C)
            if len(same_status_data) > 0:
                status_time_agg = (
                    same_status_data.groupby("time_minutes")["temperature"]
                    .mean()
                    .reindex(ts_selected["time_minutes"])
                    .interpolate(method="linear")
                    .bfill()
                    .ffill()
                )

                avg_curve = status_time_agg.values
                # Standard reefer thermostat operating tolerance (+/- 1.5°C), clamped at 0°C minimum
                upper_bound = avg_curve + 1.5
                lower_bound = np.maximum(avg_curve - 1.5, 0.0)
                time_points = ts_selected["time_minutes"].values

                # Average baseline line
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=avg_curve,
                    mode="lines",
                    name=f"Average ({selected_status})",
                    line=dict(color="#ff7f0e", width=2, dash="dash")
                ))

                # Safe Operating Tolerance Band (Filled corridor)
                fig.add_trace(go.Scatter(
                    x=time_points.tolist() + time_points.tolist()[::-1],
                    y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(255, 127, 14, 0.18)",
                    line=dict(color="rgba(255, 255, 255, 0)", width=0),
                    name="Operating Tolerance (+/- 1.5°C)",
                    hoverinfo="skip",
                    showlegend=True
                ))

            # Breach threshold line
            fig.add_hline(
                y=8.0,
                line_dash="dash",
                line_color="#2ca02c",
                annotation_text="Breach Threshold (8°C)"
            )
            fig.update_layout(
                title=f"Temperature Profile: {selected_id} (Status: {selected_status})",
                xaxis_title="Time (minutes)",
                yaxis_title="Temperature (°C)",
                height=450,
                hovermode="x unified"
            )
            st.plotly_chart(fig, width="stretch")

            # --- GPS ROUTE MAP ---
            st.subheader("GPS Route (Colored by Temperature)")
            start_point = ts_selected.iloc[0]
            end_point = ts_selected.iloc[-1]
            center_lat, center_lon = ts_selected["latitude"].mean(), ts_selected["longitude"].mean()
            
            fig_map = px.scatter_map(
                ts_selected,
                lat="latitude",
                lon="longitude",
                color="temperature",
                hover_name="time_minutes",
                hover_data={"temperature": ":.1f°C", "time_minutes": "min"},
                title=f"Route for {selected_id}",
                color_continuous_scale="RdYlBu_r",
                zoom=6,
                height=450
            )

            # Start marker (Green)
            fig_map.add_scattermap(
                lat=[start_point["latitude"]],
                lon=[start_point["longitude"]],
                mode="markers",
                marker=dict(size=14, color="#2ca02c"),
                text=["Start"],
                hoverinfo="text",
                name="Start"
            )

            # End marker (Red)
            fig_map.add_scattermap(
                lat=[end_point["latitude"]],
                lon=[end_point["longitude"]],
                mode="markers",
                marker=dict(size=14, color="#d62728"),
                text=["End"],
                hoverinfo="text",
                name="End"
            )

            fig_map.update_layout(
                map_style="open-street-map",
                map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=6)
            )
            fig_map.update_traces(marker=dict(size=8))
            st.plotly_chart(fig_map, width="stretch")

            # Breach notice
            breach_df = ts_selected[ts_selected["temperature"] > 8.0]
            if len(breach_df) > 0:
                first_breach = breach_df.iloc[0]
                breach_time = first_breach["time_minutes"]
                st.warning(
                    f"**Breach Detected**: The selected shipment breached the 8°C threshold "
                    f"at **{breach_time} minutes**.\n"
                    f"- **Location**: Latitude {first_breach['latitude']:.4f}, Longitude {first_breach['longitude']:.4f}\n"
                    f"- **Recommendation**: Check if this location corresponds to a known traffic bottleneck or delivery stop."
                )
            else:
                st.success("No breaches detected for the selected shipment.")
