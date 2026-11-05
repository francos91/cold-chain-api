import requests
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Logistics Overview", layout="wide")
st.title("Cold Chain Logistics Overview")
st.sidebar.success("Select a tool above to navigate.")

# 2. API Connection (Supabase is completely gone!)
API_URL = "https://cold-chain-api-docker.onrender.com/api/dashboard"

@st.cache_data(ttl=600)
def get_processed_data():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data["raw"]), pd.DataFrame(data["features"])
    except Exception as e:
        st.error(f"Failed to connect to backend API: {e}")
        return pd.DataFrame(), pd.DataFrame()

# 3. Sidebar Filtering
st.sidebar.header("Filter Options")
if st.sidebar.button("Refresh Data"):
  st.cache_data.clear()
  st.rerun()

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

# 4. Home Page Tabs
tab1, tab2 = st.tabs(["Overview", "Raw Data"])

with tab1:
  st.subheader("SCOR-Aligned Performance Overview")
  st.caption(f"Active Status Filter: **{', '.join(selected_status)}**")

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

  st.divider()

  st.subheader("Cost-to-Serve & Route Financial Exposure")
  col_a, col_b = st.columns(2)
  
  with col_a:
    if not filtered_df_ready.empty:
        cost_df = (
            filtered_df_ready.groupby(["origin", "destination"])
            .agg(
                avg_cost_zar=("cost_per_shipment", "mean"),
                total_shipments=("shipment_id", "count"),
                avg_transit_h=("transit_time_hours", "mean"),
            )
            .reset_index()
            .sort_values(by="avg_cost_zar", ascending=False)
            .head(5)
        )
        st.markdown("**Top 5 Highest Cost Corridors (Avg ZAR)**")
        st.dataframe(cost_df.style.format({"avg_cost_zar": "R{:,.2f}", "avg_transit_h": "{:.1f} hrs"}), width="stretch")
        
  with col_b:
    if not filtered_df_ready.empty:
        plot_data = filtered_df_ready.copy()
        plot_data["status_lower"] = plot_data["status"].str.lower()
        sample_df = plot_data.sample(n=min(500, len(plot_data)), random_state=42)
        fig_cost = px.scatter(
            sample_df,
            x="distance_km",
            y="cost_per_shipment",
            color="status_lower",
            title="Shipment Cost vs. Distance Physics",
        )
        st.plotly_chart(fig_cost, width="stretch")

with tab2:
  st.subheader("Logistics Data Table")
  st.dataframe(
      filtered_df.head(500), 
      width="stretch",
      column_config={"temperature_celsius": st.column_config.NumberColumn("Temperature (°C)", format="%.2f")},
  )
