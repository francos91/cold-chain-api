from pathlib import Path
from fastdtw import fastdtw
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import euclidean
from supabase import create_client
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Logistics Dashboard", layout="wide")
st.title("Cold Chain Logistics Dashboard")


# 2. Connection
@st.cache_resource
def init_connection():
  return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


supabase = init_connection()


# FIXED: Pagination to fetch ALL rows (no 5000 limit)
@st.cache_data(ttl=600)
def get_data():
  all_data = []
  page_size = 1000
  start = 0

  while True:
    end = start + page_size - 1
    response = (
        supabase.table("cold_chain_tracking")
        .select("*")
        .range(start, end)
        .execute()
    )

    if not response.data:
      break

    all_data.extend(response.data)
    start += page_size

    if len(response.data) < page_size:
      break

  return pd.DataFrame(all_data)


# CACHED: Temperature stream data to eliminate Tab 4 loading delay
@st.cache_data(ttl=600)
def get_temperature_stream():
  response = supabase.table("temperature_stream").select("*").execute()
  return pd.DataFrame(response.data)


# 3. Load the Advanced Model & Encoder
@st.cache_resource
def load_advanced_model():
  BASE_DIR = Path(__file__).parent
  model = joblib.load(BASE_DIR / "advanced_model.pkl")
  le_route = joblib.load(BASE_DIR / "le_route.pkl")
  return model, le_route


model, le_route = load_advanced_model()


# 4. Helper: Feature Engineering for Predictive Analytics
def engineer_features_for_app(df):
  df_feat = df.copy()
  df_feat["route"] = df_feat["origin"] + "_to_" + df_feat["destination"]

  # Calculate historical risk (mean excursions per route)
  risk_map = (
      df_feat.groupby("route")["temperature_celsius"]
      .apply(lambda x: (x > 8.0).mean())
      .to_dict()
  )
  df_feat["route_risk_score"] = df_feat["route"].map(risk_map)

  # --- MODEL FEATURES ---
  try:
    df_feat["route_enc"] = le_route.transform(df_feat["route"])
  except ValueError:
    df_feat["route_enc"] = 0

  # Delay status flag
  df_feat["is_delayed"] = (df_feat["status"] == "delayed").astype(int)
  df_feat["is_risk"] = (df_feat["temperature_celsius"] > 8.0).astype(int)

  # --- CONTINUOUS PHYSICS FEATURES ---
  df_feat["distance_km"] = pd.to_numeric(
      df_feat.get("distance_km", 500), errors="coerce"
  ).fillna(500)
  df_feat["weight_tonnes"] = pd.to_numeric(
      df_feat.get("weight_tonnes", 20.0), errors="coerce"
  ).fillna(20.0)

  if (
      "transit_time_hours" not in df_feat.columns
      or df_feat["transit_time_hours"].isnull().all()
  ):
    np.random.seed(42)
    base_transit = df_feat["distance_km"] / 65.0
    delay_penalty = df_feat["is_delayed"] * np.random.uniform(4, 12, len(df_feat))
    traffic_noise = np.random.uniform(-1, 2, len(df_feat))
    df_feat["transit_time_hours"] = (
        base_transit + delay_penalty + traffic_noise
    )
  else:
    df_feat["transit_time_hours"] = pd.to_numeric(
        df_feat["transit_time_hours"], errors="coerce"
    )

  # --- SCOR KPIs ---
  df_feat["is_ontime"] = (df_feat["status"] != "delayed").astype(int)
  df_feat["is_compliant"] = (df_feat["is_risk"] == 0).astype(int)
  df_feat["is_perfect"] = (
      (df_feat["is_ontime"] == 1) & (df_feat["is_compliant"] == 1)
  ).astype(int)

  cost_per_km = 18
  fixed_cost = 200
  df_feat["cost_per_shipment"] = (
      df_feat["distance_km"] * cost_per_km
  ) + fixed_cost

  impact_per_breach = 50000
  df_feat["vaR_ZAR"] = df_feat["route_risk_score"] * impact_per_breach
  df_feat["capacity_utilization"] = np.random.uniform(70, 90, len(df_feat))

  return df_feat


# 5. Sidebar Filtering & Cache Control
st.sidebar.header("Filter Options")

if st.sidebar.button("Refresh Data"):
  st.cache_data.clear()
  st.rerun()

df = get_data()
st.sidebar.write(f"Rows fetched from API: {len(df)}")

df_ready = engineer_features_for_app(df)

selected_status = st.sidebar.multiselect(
    "Select Status",
    options=df["status"].unique(),
    default=df["status"].unique(),
)
# FIXED: .copy() prevents SettingWithCopyWarning on downstream mutations
filtered_df = df[df["status"].isin(selected_status)].copy()
filtered_df_ready = df_ready[df_ready["status"].isin(selected_status)].copy()

# Pre-calculate ML probabilities so we don't recalculate on every slider move
X_pred = filtered_df_ready[[
    "route_enc",
    "route_risk_score",
    "is_delayed",
    "distance_km",
    "transit_time_hours",
]]
if not X_pred.empty:
    filtered_df_ready["risk_probability"] = model.predict_proba(X_pred)[:, 1]
else:
    filtered_df_ready["risk_probability"] = 0.0

# 6. Intermodal Corridor Reference Database
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

# 7. Tabs Setup
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
  st.markdown(
      "Aggregated long-haul performance across primary South African transport"
      " corridors based on live database telemetry."
  )

  if (
      not filtered_df_ready.empty
      and "distance_km" in filtered_df_ready.columns
  ):
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

    st.dataframe(
        corridor_df.style.format(
            {"avg_transit_hours": "{:.1f} hrs", "avg_distance_km": "{:.1f} km"}
        ),
        width="stretch",
    )

  st.divider()

  st.subheader("Cost-to-Serve & Route Financial Exposure")
  st.markdown(
      "Evaluating transport cost efficiency and financial exposure (VaR) across"
      " major logistics corridors."
  )

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

    col_a, col_b = st.columns(2)

    with col_a:
      st.markdown("**Top 5 Highest Cost Corridors (Avg ZAR)**")
      st.dataframe(
          cost_df.style.format(
              {"avg_cost_zar": "R{:,.2f}", "avg_transit_h": "{:.1f} hrs"}
          ),
          width="stretch",
      )

    with col_b:
      # FIXED: Normalized status to lowercase and sampled randomly to ensure delayed and in-transit points are visible
      plot_data = filtered_df_ready.copy()
      plot_data["status_lower"] = plot_data["status"].str.lower()
      sample_df = plot_data.sample(n=min(500, len(plot_data)), random_state=42)

      fig_cost = px.scatter(
          sample_df,
          x="distance_km",
          y="cost_per_shipment",
          color="status_lower",
          title="Shipment Cost vs. Distance Physics",
          labels={
              "distance_km": "Distance (km)",
              "cost_per_shipment": "Cost (ZAR)",
              "status_lower": "Status",
          },
          color_discrete_map={
              "delivered": "#008080",
              "delayed": "#FF4B4B",
              "in-transit": "#FFA500",
          },
      )
      st.plotly_chart(fig_cost, width="stretch")

  st.divider()

  st.subheader("Carbon Emissions & Financial Exposure (SCOR Cost-to-Serve)")
  st.markdown(
      "Quantifying environmental impact and carbon tax liability using ISO"
      " 14083 activity-based modelling."
  )

  DIESEL_EMISSION_FACTOR = 73.2
  GRID_BEV_EMISSION_FACTOR = 156.8
  SOLAR_BEV_EMISSION_FACTOR = 6.4
  CARBON_TAX_RATE = 28.0  # Corrected South African effective rate

  if (
      not filtered_df_ready.empty
      and "distance_km" in filtered_df_ready.columns
      and "weight_tonnes" in filtered_df_ready.columns
  ):
    filtered_df_ready["tkm"] = (
        filtered_df_ready["distance_km"] * filtered_df_ready["weight_tonnes"]
    )
    total_tkm = filtered_df_ready["tkm"].sum()

    diesel_emissions = (total_tkm * DIESEL_EMISSION_FACTOR) / 1_000_000
    grid_bev_emissions = (total_tkm * GRID_BEV_EMISSION_FACTOR) / 1_000_000
    solar_bev_emissions = (total_tkm * SOLAR_BEV_EMISSION_FACTOR) / 1_000_000

    tax_diesel = diesel_emissions * CARBON_TAX_RATE
    tax_grid = grid_bev_emissions * CARBON_TAX_RATE
    tax_solar = solar_bev_emissions * CARBON_TAX_RATE

    col1, col2, col3 = st.columns(3)

    with col1:
      st.metric(
          label="Current Fleet (Diesel Baseline)",
          value=f"{diesel_emissions:,.2f} tCO2",
          delta=f"R {tax_diesel:,.2f} Tax Liability",
          delta_color="inverse",
      )
    with col2:
      st.metric(
          label="Green Fleet (Solar BEV)",
          value=f"{solar_bev_emissions:,.2f} tCO2",
          delta=f"R {tax_solar:,.2f} Tax Liability",
          delta_color="normal",
      )
    with col3:
      st.metric(
          label="Grid BEV (Eskom Coal Mix Warning)",
          value=f"{grid_bev_emissions:,.2f} tCO2",
          delta=f"R {tax_grid:,.2f} Tax Liability",
          delta_color="inverse",
      )

    st.info(
        "Sustainable Logistics Insight: Calculations utilize dynamic individual"
        " shipment payload masses and ISO 14083 activity-based tkm methodology."
    )
  else:
    st.warning("Required distance or weight data not available in the current view.")

  st.divider()

  st.subheader("Temperature Distribution")
  fig = px.histogram(
      filtered_df,
      x="temperature_celsius",
      nbins=20,
      color_discrete_sequence=["#008080"],
  )
  st.plotly_chart(fig, width="stretch")

  with st.expander("SCOR Framework Alignment"):
    st.markdown("""
        **Reliability (RL.1.1)** - Perfect Order Fulfillment (On-Time + No Breach)  
        **Responsiveness (RS.1.1)** - Average Transit Time (hours)  
        **Agility (AG.1.3)** - Value at Risk (route_risk_score x impact)  
        **Cost (CO.3.15)** - Cost per Shipment (distance x rand/km + fixed)  
        **Asset Management (AM.3.9)** - Capacity Utilization (simulated)  
        """)

with tab2:
  st.subheader("Logistics Data Table")
  st.caption(f"Displaying rows for status: **{', '.join(selected_status)}**")
  
  # MOBILE FIX: Limit rows to 500 so the browser doesn't crash
  st.dataframe(
      filtered_df.head(500), 
      width="stretch",
      column_config={
          "temperature_celsius": st.column_config.NumberColumn(
              "Temperature (°C)", format="%.2f"
          )
      },
  )
  if len(filtered_df) > 500:
      st.caption("Showing the first 500 rows for mobile performance. Use the sidebar to filter.")

with tab3:
  st.subheader("Predictive Risk Intelligence")
  st.info(
      f"Active View Selections | Status Filter: `{', '.join(selected_status)}` |"
      f" Total Shipments Evaluated: `{len(filtered_df_ready):,}`"
  )

  @st.fragment
  def render_risk_fragment(data):
      # FIXED: .copy() prevents SettingWithCopyWarning on data["predicted_risk"]
      data = data.copy()

      threshold_pct = st.slider(
          "Risk Threshold (%)",
          min_value=5,
          max_value=95,
          value=20,
          step=5,
          format="%d%%",
          help=(
              "Lower threshold = catches more breaches (higher recall), but may"
              " increase false alarms (lower precision)."
          ),
      )
    
      threshold = threshold_pct / 100.0
    
      st.caption(
          f"Currently using a **{threshold_pct}%** threshold. Shipments with a"
          " probability above this are flagged as high-risk."
      )
    
      data["predicted_risk"] = (
          data["risk_probability"] >= threshold
      ).astype(int)
    
      high_risk_count = int(data["predicted_risk"].sum())
      total_count = len(data)
      high_risk_share = (
          (high_risk_count / total_count) * 100 if total_count > 0 else 0
      )
      max_model_prob = (
          data["risk_probability"].max() * 100
          if len(data) > 0
          else 0
      )
    
      col1, col2 = st.columns(2)
      col1.metric("Predicted High-Risk Shipments", high_risk_count)
      col2.metric("High-Risk Fleet Share", f"{high_risk_share:.1f}%")
    
      st.caption(
          "Model max probability score in current view:"
          f" **{max_model_prob:.1f}%**. If the threshold is set higher than this,"
          " risk count will be zero."
      )
    
      st.divider()
    
      st.markdown("### Top 5 Highest Risk Shipments")
    
      high_risk_df = data[
          (data["predicted_risk"] == 1)
          & (data["status"] != "delivered")
      ]
    
      top_5_df = high_risk_df.sort_values(
          "risk_probability", ascending=False
      ).head(5)
    
      if len(top_5_df) > 0:
        display_cols = [
            "shipment_id",
            "origin",
            "destination",
            "temperature_celsius",
            "status",
            "risk_probability",
        ]
        st.dataframe(
            top_5_df[display_cols],
            width="stretch",
            column_config={
                "shipment_id": "Shipment ID",
                "origin": "Origin",
                "destination": "Destination",
                "temperature_celsius": st.column_config.NumberColumn(
                    "Temperature (°C)", format="%.2f"
                ),
                "status": "Status",
                "risk_probability": st.column_config.NumberColumn(
                    "Risk Probability", format="%.1f%%"
                ),
            },
        )
    
        highest_risk = top_5_df.iloc[0]
        st.metric(
            f"Highest Risk: {highest_risk['shipment_id']}",
            f"{highest_risk['risk_probability']:.1%}",
            delta=f"{highest_risk['origin']} -> {highest_risk['destination']}",
        )
      else:
        st.success(
            "No active high-risk shipments predicted at the"
            f" **{threshold_pct}%** threshold!"
        )
    
      st.divider()
    
      st.markdown("### Full High-Risk Shipments List")
    
      if len(high_risk_df) > 0:
        display_cols = [
            "shipment_id",
            "origin",
            "destination",
            "temperature_celsius",
            "status",
            "risk_probability",
        ]
        st.dataframe(
            high_risk_df[display_cols],
            width="stretch",
            column_config={
                "shipment_id": "Shipment ID",
                "origin": "Origin",
                "destination": "Destination",
                "temperature_celsius": st.column_config.NumberColumn(
                    "Temperature (°C)", format="%.2f"
                ),
                "status": "Status",
                "risk_probability": st.column_config.NumberColumn(
                    "Risk Probability", format="%.1f%%"
                ),
            },
        )
        st.caption(
            "Showing "
            f"{len(high_risk_df)} active high-risk shipments at the"
            f" **{threshold_pct}%** threshold."
        )
      else:
        st.success(
            f"No active high-risk shipments at the **{threshold_pct}%** threshold!"
        )

  # Call the fragment function
  render_risk_fragment(filtered_df_ready)

  st.markdown("### High-Risk Routes Analysis")
  route_risk = (
      filtered_df_ready.groupby("route")["is_risk"].mean().reset_index()
  )
  fig_risk = px.bar(
      route_risk.sort_values("is_risk", ascending=False).head(10),
      x="is_risk",
      y="route",
      orientation="h",
      title="Routes with Highest Excursion Frequency",
  )
  st.plotly_chart(fig_risk, width="stretch")

  st.info(
      "The model analysis indicates that spatial routing characteristics"
      " (route_risk_score) are the primary drivers of cold chain breaches,"
      " providing a data-driven basis for infrastructure investment in"
      " high-risk corridors."
  )

with tab4:
  st.subheader("Thermal Profile Analysis")
  st.markdown(
      "View the temperature curve and GPS route for a shipment, compared to the"
      " average for its status."
  )
  st.caption(f"Active Status Filter: **{', '.join(selected_status)}**")

  # MOBILE FIX: Lazy-load the 66,000 rows and the heavy GPS map
  if "thermal_loaded" not in st.session_state:
    st.session_state.thermal_loaded = False

  if st.button("Load Thermal Telemetry & GPS Map") or st.session_state.thermal_loaded:
    st.session_state.thermal_loaded = True

    with st.spinner("Loading thermal profile data..."):
      ts_df = get_temperature_stream()

    if len(ts_df) == 0:
      st.warning("No time-series data found. Please run the generator first.")
    elif len(selected_status) == 0:
      st.info("No status selected in the sidebar.")
    else:
      status_resp = (
          supabase.table("cold_chain_tracking")
          .select("shipment_id")
          .in_("status", selected_status)
          .execute()
      )
      filtered_shipment_ids = [row["shipment_id"] for row in status_resp.data]

      ts_df_filtered = ts_df[ts_df["shipment_id"].isin(filtered_shipment_ids)]

      if len(ts_df_filtered) == 0:
        st.info(f"No time-series data found for status(es): {', '.join(selected_status)}")
      else:
        shipment_ids = ts_df_filtered["shipment_id"].unique()
        selected_id = st.selectbox("Select Shipment to Analyze", shipment_ids, index=0)

        ts_selected = (
            ts_df_filtered[ts_df_filtered["shipment_id"] == selected_id]
            .sort_values("time_minutes")
        )

        if len(ts_selected) == 0:
          st.warning("No temperature data found for this shipment.")
        else:
          status_resp_single = (
              supabase.table("cold_chain_tracking")
              .select("status")
              .eq("shipment_id", selected_id)
              .execute()
          )
          shipment_status = status_resp_single.data[0]["status"]

          st.subheader("Quick Summary")
          col1, col2, col3 = st.columns(3)
          col1.metric("Shipment", selected_id)
          col2.metric("Status", shipment_status.capitalize())
          col3.metric("Avg Temp", f"{ts_selected['temperature'].mean():.1f}°C")

          st.subheader("Comparison to Average")

          status_resp_all = (
              supabase.table("cold_chain_tracking")
              .select("shipment_id")
              .eq("status", shipment_status)
              .execute()
          )
          same_status_ids = [row["shipment_id"] for row in status_resp_all.data]

          same_status_data = ts_df_filtered[
              ts_df_filtered["shipment_id"].isin(same_status_ids)
          ]

          if len(same_status_data) > 0:
            avg_per_shipment = same_status_data.groupby("shipment_id")[
                "temperature"
            ].mean()
            status_avg_temp = avg_per_shipment.mean()
            status_std_temp = avg_per_shipment.std()
            selected_avg_temp = ts_selected["temperature"].mean()
            temp_diff = selected_avg_temp - status_avg_temp

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Status Average ({shipment_status})", f"{status_avg_temp:.1f}°C")
            col2.metric("Your Shipment", f"{selected_avg_temp:.1f}°C")

            if temp_diff > 0:
              col3.metric("Difference", f"+{temp_diff:.1f}°C", delta="Warmer than average", delta_color="inverse")
              st.warning(f"This shipment is **{temp_diff:.1f}°C warmer** than the average {shipment_status} shipment.")
            else:
              col3.metric("Difference", f"{temp_diff:.1f}°C", delta="Cooler than average", delta_color="normal")
              st.success(f"This shipment is **{abs(temp_diff):.1f}°C cooler** than the average {shipment_status} shipment.")

            breach_count_same = same_status_data[same_status_data["temperature"] > 8.0]["shipment_id"].nunique()
            total_same = len(avg_per_shipment)
            if total_same > 0:
              breach_rate = (breach_count_same / total_same) * 100
              st.info(f"**{breach_rate:.1f}%** of {shipment_status} shipments had at least one temperature breach (>8°C).")

          st.subheader("Temperature Profile")
          fig = go.Figure()

          fig.add_trace(
              go.Scatter(
                  x=ts_selected["time_minutes"],
                  y=ts_selected["temperature"],
                  mode="lines+markers",
                  name=f"{selected_id} ({shipment_status})",
                  line=dict(color="blue", width=3),
              )
          )

          if len(same_status_data) > 0:
            time_points = ts_selected["time_minutes"].values
            all_curves = []
            for sid in avg_per_shipment.index[:10]:
              curve = (
                  ts_df_filtered[ts_df_filtered["shipment_id"] == sid]
                  .sort_values("time_minutes")["temperature"]
                  .values
              )
              if len(curve) == len(time_points):
                all_curves.append(curve)

            if len(all_curves) > 0:
              avg_curve = np.mean(all_curves, axis=0)
              std_curve = np.std(all_curves, axis=0)

              fig.add_trace(
                  go.Scatter(
                      x=time_points,
                      y=avg_curve,
                      mode="lines",
                      name=f"Average ({shipment_status})",
                      line=dict(color="orange", width=2, dash="dash"),
                  )
              )

              fig.add_trace(
                  go.Scatter(
                      x=time_points.tolist() + time_points.tolist()[::-1],
                      y=(avg_curve + std_curve).tolist()
                      + (avg_curve - std_curve).tolist()[::-1],
                      fill="toself",
                      fillcolor="rgba(255, 165, 0, 0.2)",
                      line=dict(color="rgba(255, 255, 255, 0)"),
                      name="+/- 1 Std Dev",
                      showlegend=True,
                  )
              )

          fig.add_hline(
              y=8.0,
              line_dash="dash",
              line_color="green",
              annotation_text="Breach Threshold (8°C)",
          )
          fig.update_layout(
              title=f"Temperature Profile: {selected_id} (Status: {shipment_status})",
              xaxis_title="Time (minutes)",
              yaxis_title="Temperature (°C)",
              height=450,
              hovermode="x unified",
          )
          st.plotly_chart(fig, width="stretch")

          st.subheader("GPS Route (Colored by Temperature)")

          start_point = ts_selected.iloc[0]
          end_point = ts_selected.iloc[-1]

          center_lat = ts_selected["latitude"].mean()
          center_lon = ts_selected["longitude"].mean()

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
              height=450,
          )

          fig_map.add_scattermap(
              lat=[start_point["latitude"]],
              lon=[start_point["longitude"]],
              mode="markers",
              marker=dict(size=14, color="green"),
              text=["Start"],
              hoverinfo="text",
              name="Start",
          )

          fig_map.add_scattermap(
              lat=[end_point["latitude"]],
              lon=[end_point["longitude"]],
              mode="markers",
              marker=dict(size=14, color="red"),
              text=["End"],
              hoverinfo="text",
              name="End",
          )

          fig_map.update_layout(
              map_style="open-street-map",
              map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=6),
          )
          fig_map.update_traces(marker=dict(size=8))
          st.plotly_chart(fig_map, width="stretch")

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

# =============================================
# TAB 5: INTERMODAL WHAT-IF SIMULATOR
# =============================================
with tab5:
  st.subheader("Intermodal Mode-Switching")
  st.markdown(
      "Evaluate the carbon, tax, and cost-to-serve trade-offs of shifting"
      " long-haul refrigerated freight from road corridors (N3/N1) to Transnet"
      " Freight Rail (TFR)."
  )

  @st.fragment
  def render_intermodal_fragment(data, corridor_data):
      col_ctrl1, col_ctrl2 = st.columns(2)
      with col_ctrl1:
        selected_corridor = st.selectbox(
            "Select Strategic Corridor", options=list(corridor_data.keys())
        )
      with col_ctrl2:
        shift_percentage = st.slider(
            "Percentage of Freight Shifted from Road to Rail (%)",
            0,
            100,
            30,
            step=5,
        )

      params = corridor_data[selected_corridor]

      if not data.empty and "distance_km" in data.columns:
        total_corridor_tkm = (
            data["distance_km"] * data["weight_tonnes"]
        ).sum()
        total_shipments = len(data)
      else:
        total_corridor_tkm = 580000.0
        total_shipments = 45

      baseline_emissions_tonnes = (
          total_corridor_tkm * params["road_emission_factor"]
      ) / 1_000_000
      baseline_cost = (
          total_corridor_tkm * (params["road_cost_per_km"] / 50.0)
      ) + (total_shipments * 2500)

      rail_tkm = total_corridor_tkm * (shift_percentage / 100.0)
      remaining_road_tkm = total_corridor_tkm * (
          1 - (shift_percentage / 100.0)
      )

      new_emissions_tonnes = (
          (remaining_road_tkm * params["road_emission_factor"])
          + (rail_tkm * params["rail_emission_factor"])
      ) / 1_000_000

      carbon_savings_tonnes = baseline_emissions_tonnes - new_emissions_tonnes
      carbon_tax_savings_zar = carbon_savings_tonnes * 28.0  # Corrected South African effective rate

      avg_road_hours = params["road_distance_km"] / params["avg_road_speed_kmh"]
      avg_rail_hours = (
          params["rail_distance_km"] / params["avg_rail_speed_kmh"]
      ) + 18.0
      transit_time_delta_hrs = (shift_percentage / 100.0) * (
          avg_rail_hours - avg_road_hours
      )

      col_m1, col_m2, col_m3, col_m4 = st.columns(4)
      col_m1.metric(
          "Carbon Emissions Delta",
          f"-{carbon_savings_tonnes:.2f} tCO2",
          delta=f"{shift_percentage}% Rail Shift",
      )
      col_m2.metric(
          "Carbon Tax Liability Saved",
          f"R {carbon_tax_savings_zar:,.2f}",
          delta="Financial Benefit",
          delta_color="normal",
      )
      col_m3.metric(
          "Modal Distance Factor",
          f"{params['rail_distance_km']} km",
          delta=f"Road: {params['road_distance_km']} km",
          delta_color="off",
      )
      col_m4.metric(
          "Transit Time Impact",
          f"+{transit_time_delta_hrs:.1f} hrs",
          delta="Terminal Delay Penalty",
          delta_color="inverse",
      )

      st.markdown("### Modal Performance Comparison")
      comparison_df = pd.DataFrame({
          "Metric": [
              "Carbon Footprint (tCO2e)",
              "Estimated Cost Index (ZAR)",
              "Average Transit Time (Hours)",
          ],
          "Baseline (100% Road)": [
              baseline_emissions_tonnes,
              baseline_cost,
              avg_road_hours,
          ],
          f"Simulated ({shift_percentage}% Rail)": [
              new_emissions_tonnes,
              # FIXED: cost now scales smoothly from 100% (0% rail) to 92% (100% rail)
              baseline_cost * (1 - (shift_percentage / 100.0) * 0.08),
              avg_road_hours
              + ((shift_percentage / 100.0) * max(0, transit_time_delta_hrs)),
          ],
      })
      st.dataframe(
          comparison_df.set_index("Metric"), width="stretch"
      )

  render_intermodal_fragment(filtered_df_ready, INTERMODAL_CORRIDOR_DATA)
