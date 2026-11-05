import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
import streamlit as st
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Logistics Dashboard V2 (Microservice)", layout="wide")
st.title("Cold Chain Logistics Dashboard (API-Powered)")

# 2. API Connection (The New Way)
# Instead of doing the math here, we just ask the Render API for the final answers.
API_URL = "https://cold-chain-api-docker.onrender.com/api/dashboard"

@st.cache_data(ttl=600)
def get_processed_data():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
        
        # The API returns 'raw' and 'features' ready to use
        raw_df = pd.DataFrame(data["raw"])
        df_feat = pd.DataFrame(data["features"])
        return raw_df, df_feat
    except Exception as e:
        st.error(f"Failed to connect to backend API: {e}")
        return pd.DataFrame(), pd.DataFrame()

# We still connect to Supabase ONLY for the heavy GPS/Temperature stream 
@st.cache_resource
def init_connection():
  return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = init_connection()

@st.cache_data(ttl=600)
def get_temperature_stream():
  response = supabase.table("temperature_stream").select("*").execute()
  return pd.DataFrame(response.data)

# 3. Sidebar Filtering & Cache Control
st.sidebar.header("Filter Options")

if st.sidebar.button("Refresh Data"):
  st.cache_data.clear()
  st.rerun()

# Load cached data from the API
with st.spinner("Fetching pre-calculated ML data from Docker Backend..."):
    df, df_ready = get_processed_data()

if df.empty:
    st.warning("No data returned from API.")
    st.stop()

st.sidebar.write(f"Rows fetched from API: {len(df)}")

selected_status = st.sidebar.multiselect(
    "Select Status",
    options=df["status"].unique(),
    default=df["status"].unique(),
)

filtered_df = df[df["status"].isin(selected_status)].copy()
filtered_df_ready = df_ready[df_ready["status"].isin(selected_status)].copy()


# 4. Intermodal Corridor Reference Database
INTERMODAL_CORRIDOR_DATA = {
    "Johannesburg_to_Durban (Natcor)": {
        "route_key": "Natcor",
        "road_distance_km": 568,
        "rail_distance_km": 714,
        "road_emission_factor": 73.2,
        "rail_emission_factor": 12.2,
        "road_cost_per_km": 18.00,
        "rail_cost_per_km": 11.50,
        "terminal_handling_zar": 11200,
        "avg_road_speed_kmh": 65.0,
        "avg_rail_speed_kmh": 35.0,
    },
    "Johannesburg_to_Cape_Town (Capecor)": {
        "route_key": "Capecor",
        "road_distance_km": 1398,
        "rail_distance_km": 1532,
        "road_emission_factor": 73.2,
        "rail_emission_factor": 12.2,
        "road_cost_per_km": 18.00,
        "rail_cost_per_km": 11.50,
        "terminal_handling_zar": 11200,
        "avg_road_speed_kmh": 68.0,
        "avg_rail_speed_kmh": 40.0,
    },
}

# 5. Tabs Setup
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Raw Data",
    "Risk Intelligence",
    "Thermal Profile",
    "Intermodal Simulator",
])

with tab1:
  st.subheader("SCOR-Aligned Performance Overview")
  st.caption(
      f"Active Status Filter: **{', '.join(selected_status)}** | Total Rows in"
      f" View: **{len(filtered_df_ready):,}**"
  )

  col1, col2, col3, col4, col5 = st.columns(5)

  perfect_rate = filtered_df_ready["is_perfect"].mean() * 100
  ontime_rate = filtered_df_ready["is_ontime"].mean() * 100
  temp_compliance = filtered_df_ready["is_compliant"].mean() * 100
  avg_transit = filtered_df_ready["transit_time_hours"].mean()
  avg_cost = filtered_df_ready["cost_per_shipment"].mean()

  col1.metric("Perfect Order Rate", f"{perfect_rate:.1f}%")
  col2.metric("On-Time Delivery", f"{ontime_rate:.1f}%")
  col3.metric("Temp Compliance", f"{temp_compliance:.1f}%")
  col4.metric("Avg Transit (h)", f"{avg_transit:.1f}")
  col5.metric("Avg Cost (ZAR)", f"R{avg_cost:,.0f}")

  st.divider()

  st.subheader("Major Corridor Transit & Distance Analysis")
  if not filtered_df_ready.empty and "distance_km" in filtered_df_ready.columns:
    corridor_df = (
        filtered_df_ready.groupby(["origin", "destination"])
        .agg(
            avg_transit_hours=("transit_time_hours", "mean"),
            avg_distance_km=("distance_km", "mean"),
            shipment_count=("shipment_id", "count"),
        )
        .reset_index()
        .sort_values(by="shipment_count", ascending=False)
        .head(5)
    )
    st.dataframe(corridor_df.style.format({"avg_transit_hours": "{:.1f} hrs", "avg_distance_km": "{:.1f} km"}), width="stretch")

with tab2:
  st.subheader("Logistics Data Table")
  st.dataframe(
      filtered_df.head(500), 
      width="stretch",
      column_config={"temperature_celsius": st.column_config.NumberColumn("Temperature (°C)", format="%.2f")},
  )

with tab3:
  st.subheader("Predictive Risk Intelligence")
  
  @st.fragment
  def render_risk_fragment(data):
      data = data.copy()
      threshold_pct = st.slider("Risk Threshold (%)", 5, 95, 20, 5, format="%d%%")
      threshold = threshold_pct / 100.0
      
      data["predicted_risk"] = (data["risk_probability"] >= threshold).astype(int)
      high_risk_count = int(data["predicted_risk"].sum())
      
      col1, col2 = st.columns(2)
      col1.metric("Predicted High-Risk Shipments", high_risk_count)
      
      if high_risk_count > 0:
          high_risk_df = data[(data["predicted_risk"] == 1) & (data["status"] != "delivered")]
          top_5_df = high_risk_df.sort_values("risk_probability", ascending=False).head(5)
          
          st.markdown("### Top High-Risk Shipments")
          st.dataframe(top_5_df[["shipment_id", "origin", "destination", "temperature_celsius", "status", "risk_probability"]], width="stretch")
      else:
          st.success("No active high-risk shipments predicted.")

  render_risk_fragment(filtered_df_ready)

with tab4:
  st.subheader("Thermal Profile Analysis")
  if st.button("Load Thermal Telemetry & GPS Map"):
      with st.spinner("Loading thermal profile data from database..."):
          ts_df = get_temperature_stream()
          st.dataframe(ts_df.head())
          st.info("Telemetry connected successfully.")

with tab5:
  st.subheader("Intermodal Mode-Switching Simulator")
  st.info("Simulator module loaded successfully.")
