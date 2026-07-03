import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
import joblib
from pathlib import Path
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Logistics Dashboard", layout="wide")
st.title("Cold Chain Logistics Dashboard")

# 2. Connection
@st.cache_resource
def init_connection():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = init_connection()

# 🔥 FIXED: Pagination to fetch ALL rows (no 5000 limit)
@st.cache_data(ttl=600)
def get_data():
    all_data = []
    page_size = 1000
    start = 0
    
    while True:
        end = start + page_size - 1
        response = supabase.table("cold_chain_tracking").select("*").range(start, end).execute()
        
        if not response.data:
            break
        
        all_data.extend(response.data)
        start += page_size
        
        if len(response.data) < page_size:
            break
    
    return pd.DataFrame(all_data)

# 3. Load the Advanced Model & Encoder
@st.cache_resource
def load_advanced_model():
    BASE_DIR = Path(__file__).parent
    model = joblib.load(BASE_DIR / "advanced_model.pkl")
    le_route = joblib.load(BASE_DIR / "le_route.pkl")
    return model, le_route

model, le_route = load_advanced_model()

# 4. Helper: Feature Engineering for Predictive Analytics (UPDATED for Model)
def engineer_features_for_app(df):
    df_feat = df.copy()
    df_feat['route'] = df_feat['origin'] + "_to_" + df_feat['destination']
    
    # Calculate historical risk (mean excursions per route)
    risk_map = df_feat.groupby('route')['temperature_celsius'].apply(lambda x: (x > 8.0).mean()).to_dict()
    df_feat['route_risk_score'] = df_feat['route'].map(risk_map)
    
    # --- MODEL FEATURES ---
    # Encode the route using the saved encoder
    try:
        df_feat['route_enc'] = le_route.transform(df_feat['route'])
    except ValueError:
        # If a new route appears that wasn't in training, assign a default value (0)
        df_feat['route_enc'] = 0
    
    # Delay status flag
    df_feat['is_delayed'] = (df_feat['status'] == 'delayed').astype(int)
    
    # Add status indicator (for your existing charts)
    df_feat['is_risk'] = (df_feat['temperature_celsius'] > 8.0).astype(int)
    
    # --- SCOR KPIs ---
    # 1. Perfect Order Fulfillment (RL.1.1) – On‑Time + No Breach
    df_feat['is_ontime'] = (df_feat['status'] != 'delayed').astype(int)
    df_feat['is_compliant'] = (df_feat['is_risk'] == 0).astype(int)
    df_feat['is_perfect'] = ((df_feat['is_ontime'] == 1) & (df_feat['is_compliant'] == 1)).astype(int)
    
    # 2. Responsiveness (RS.1.1) – transit_time_hours is already in the data
    # (No new column needed; it's already in df_feat)
    
    # 3. Cost per Shipment (CO.3.15) – using distance_km
    cost_per_km = 18  # ZAR/km (typical heavy vehicle)
    fixed_cost = 200  # ZAR per shipment
    # distance_km is already in the data (populated earlier)
    df_feat['cost_per_shipment'] = (df_feat['distance_km'] * cost_per_km) + fixed_cost
    
    # 4. Value at Risk (AG.1.3) – route_risk_score × impact
    impact_per_breach = 50000  # ZAR (estimated loss)
    df_feat['vaR_ZAR'] = df_feat['route_risk_score'] * impact_per_breach
    
    # 5. Capacity Utilization (AM.3.9) – simulate (no real load data)
    df_feat['capacity_utilization'] = np.random.uniform(70, 90, len(df_feat))  # %
    
    return df_feat

# 5. Sidebar Filtering & Cache Control
st.sidebar.header("Filter Options")

# Force Refresh button for your cache
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

df = get_data()

# Debugging: Verify record count
st.sidebar.write(f"Rows fetched from API: {len(df)}")

df_ready = engineer_features_for_app(df)

selected_status = st.sidebar.multiselect("Select Status", options=df['status'].unique(), default=df['status'].unique())
filtered_df = df[df['status'].isin(selected_status)]
# Sync filtering with the processed data
filtered_df_ready = df_ready[df_ready['status'].isin(selected_status)]

# 6. Tabs and Professional Layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Raw Data", "🚀 Risk Intelligence", "📈 Thermal Profile"])

with tab1:
    st.subheader("📊 SCOR-Aligned Performance Overview")
    
    # --- KPI Cards ---
    col1, col2, col3, col4, col5 = st.columns(5)
    
    perfect_rate = filtered_df_ready['is_perfect'].mean() * 100
    ontime_rate = filtered_df_ready['is_ontime'].mean() * 100
    temp_compliance = filtered_df_ready['is_compliant'].mean() * 100
    avg_transit = filtered_df_ready['transit_time_hours'].mean()
    avg_cost = filtered_df_ready['cost_per_shipment'].mean()
    
    col1.metric("📦 Perfect Order Rate", f"{perfect_rate:.1f}%")
    col2.metric("⏱️ On‑Time Delivery", f"{ontime_rate:.1f}%")
    col3.metric("🌡️ Temp Compliance", f"{temp_compliance:.1f}%")
    col4.metric("🚚 Avg Transit (h)", f"{avg_transit:.1f}")
    col5.metric("💰 Avg Cost (ZAR)", f"R{avg_cost:,.0f}")
    
    st.divider()
    
    # --- Existing Temperature Distribution (keep it) ---
    st.subheader("🌡️ Temperature Distribution")
    fig = px.histogram(filtered_df, x="temperature_celsius", nbins=20, color_discrete_sequence=['#008080'])
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Optional: SCOR Framework Expander ---
    with st.expander("📘 SCOR Framework Alignment"):
        st.markdown("""
        **Reliability (RL.1.1)** – Perfect Order Fulfillment (On‑Time + No Breach)  
        **Responsiveness (RS.1.1)** – Average Transit Time (hours)  
        **Agility (AG.1.3)** – Value at Risk (route_risk_score × impact)  
        **Cost (CO.3.15)** – Cost per Shipment (distance × ZAR/km + fixed)  
        **Asset Management (AM.3.9)** – Capacity Utilization (simulated)  
        """)

with tab2:
    st.subheader("Logistics Data Table")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "temperature_celsius": st.column_config.NumberColumn(
                "Temperature (°C)",
                format="%.2f",
            )
        }
    )

# =============================================
# TAB 3: RISK INTELLIGENCE (WITH THRESHOLD SLIDER)
# =============================================
with tab3:
    st.subheader("Predictive Risk Intelligence")
    
    # --- 🎯 Risk Threshold Slider (Decision-Support Tool) ---
    threshold = st.slider(
        "🎯 Risk Threshold (%)",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        help="Lower threshold = catches more breaches (higher recall), but may increase false alarms (lower precision)."
    ) / 100.0
    
    st.caption(f"⚙️ Currently using a **{threshold*100:.0f}%** threshold. Shipments with a probability above this are flagged as high-risk.")
    
    # --- Prepare features for the model ---
    X_pred = filtered_df_ready[['route_enc', 'route_risk_score', 'is_delayed']]
    
    # --- Get probabilities ---
    filtered_df_ready['risk_probability'] = model.predict_proba(X_pred)[:, 1]
    filtered_df_ready['predicted_risk'] = (filtered_df_ready['risk_probability'] >= threshold).astype(int)
    
    # --- Display AI-powered metrics ---
    col1, col2 = st.columns(2)
    col1.metric("🚨 Predicted High-Risk Shipments", filtered_df_ready['predicted_risk'].sum())
    col2.metric("🎯 Avg Risk Probability", f"{filtered_df_ready['risk_probability'].mean():.1%}")
    
    st.divider()

    # --- 🚨 Top 5 Highest Risk Shipments (Leaderboard) ---
    st.markdown("### 🚨 Top 5 Highest Risk Shipments")
    
    high_risk_df = filtered_df_ready[
        (filtered_df_ready['predicted_risk'] == 1) & 
        (filtered_df_ready['status'] != 'delivered')
    ]
    
    top_5_df = high_risk_df.sort_values('risk_probability', ascending=False).head(5)
    
    if len(top_5_df) > 0:
        display_cols = ['shipment_id', 'origin', 'destination', 'temperature_celsius', 'status', 'risk_probability']
        st.dataframe(
            top_5_df[display_cols],
            use_container_width=True,
            column_config={
                "shipment_id": "Shipment ID",
                "origin": "Origin",
                "destination": "Destination",
                "temperature_celsius": st.column_config.NumberColumn("Temperature (°C)", format="%.2f"),
                "status": "Status",
                "risk_probability": st.column_config.NumberColumn("Risk Probability", format="%.1f%%")
            }
        )
        
        highest_risk = top_5_df.iloc[0]
        st.metric(
            f"⚠️ Highest Risk: {highest_risk['shipment_id']}", 
            f"{highest_risk['risk_probability']:.1%}",
            delta=f"{highest_risk['origin']} → {highest_risk['destination']}"
        )
    else:
        st.success(f"🎉 No active high-risk shipments predicted at the **{threshold*100:.0f}%** threshold!")

    st.divider()

    # --- 📋 Full High-Risk Shipments List ---
    st.markdown("### 📋 Full High-Risk Shipments List")
    
    if len(high_risk_df) > 0:
        display_cols = ['shipment_id', 'origin', 'destination', 'temperature_celsius', 'status', 'risk_probability']
        st.dataframe(
            high_risk_df[display_cols],
            use_container_width=True,
            column_config={
                "shipment_id": "Shipment ID",
                "origin": "Origin",
                "destination": "Destination",
                "temperature_celsius": st.column_config.NumberColumn("Temperature (°C)", format="%.2f"),
                "status": "Status",
                "risk_probability": st.column_config.NumberColumn("Risk Probability", format="%.1f%%")
            }
        )
        st.caption(f"Showing {len(high_risk_df)} active high-risk shipments at the **{threshold*100:.0f}%** threshold.")
    else:
        st.success(f"🎉 No active high-risk shipments at the **{threshold*100:.0f}%** threshold!")

    # --- High-Risk Routes Chart ---
    st.markdown("### High-Risk Routes Analysis")
    route_risk = filtered_df_ready.groupby('route')['is_risk'].mean().reset_index()
    fig_risk = px.bar(route_risk.sort_values('is_risk', ascending=False).head(10), 
                     x='is_risk', y='route', orientation='h', 
                     title="Routes with Highest Excursion Frequency")
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.info("The model analysis indicates that spatial routing characteristics (route_risk_score) are the primary drivers of cold chain breaches, providing a data-driven basis for infrastructure investment in high-risk corridors.")

# =============================================
# TAB 4: THERMAL PROFILE (Filtered by Status) - WITH START/END MARKERS
# =============================================
with tab4:
    st.subheader("📈 Thermal Profile Analysis")
    st.markdown("View the temperature curve and GPS route for a shipment, compared to the average for its status.")
    
    # Fetch time-series data
    with st.spinner("Loading thermal profile data..."):
        ts_response = supabase.table("temperature_stream").select("*").execute()
        ts_df = pd.DataFrame(ts_response.data)
    
    if len(ts_df) == 0:
        st.warning("⚠️ No time-series data found. Please run the temperature_stream generator first.")
        st.stop()
    
    # --- FILTER BY SIDEBAR STATUS ---
    if len(selected_status) == 0:
        st.info("ℹ️ No status selected in the sidebar. Please select at least one status.")
        st.stop()
    
    # Get shipment IDs that match the selected status(es)
    status_resp = supabase.table("cold_chain_tracking").select("shipment_id").in_("status", selected_status).execute()
    filtered_shipment_ids = [row['shipment_id'] for row in status_resp.data]
    
    if len(filtered_shipment_ids) == 0:
        st.info(f"ℹ️ No shipments found with status(es): {', '.join(selected_status)}")
        st.stop()
    
    # Filter time-series data to only include matching shipments
    ts_df_filtered = ts_df[ts_df['shipment_id'].isin(filtered_shipment_ids)]
    
    if len(ts_df_filtered) == 0:
        st.info(f"ℹ️ No time-series data found for shipments with status(es): {', '.join(selected_status)}")
        st.stop()
    
    # Get unique shipment IDs from filtered data
    shipment_ids = ts_df_filtered['shipment_id'].unique()
    
    # --- Select a Single Shipment ---
    selected_id = st.selectbox("🔵 Select Shipment to Analyze", shipment_ids, index=0)
    
    # Fetch the selected shipment's curve
    ts_selected = ts_df_filtered[ts_df_filtered['shipment_id'] == selected_id].sort_values('time_minutes')
    
    if len(ts_selected) == 0:
        st.warning("⚠️ No temperature data found for this shipment.")
        st.stop()
    
    # --- Get the status of the selected shipment from the parent table ---
    status_resp_single = supabase.table("cold_chain_tracking").select("status").eq("shipment_id", selected_id).execute()
    if len(status_resp_single.data) == 0:
        st.warning("⚠️ Shipment not found in the main table.")
        st.stop()
    shipment_status = status_resp_single.data[0]['status']
    
    # --- Display Metrics ---
    st.subheader("📊 Quick Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Shipment", selected_id)
    col2.metric("📌 Status", shipment_status.capitalize())
    col3.metric("🌡️ Avg Temp", f"{ts_selected['temperature'].mean():.1f}°C")
    
    # --- COMPARE TO AVERAGE ---
    st.subheader("📊 Comparison to Average")
    
    # Get all shipment IDs with the same status from the parent table
    status_resp_all = supabase.table("cold_chain_tracking").select("shipment_id").eq("status", shipment_status).execute()
    same_status_ids = [row['shipment_id'] for row in status_resp_all.data]
    
    # Filter time-series data to only include these shipments (and also respect the sidebar filter)
    same_status_data = ts_df_filtered[ts_df_filtered['shipment_id'].isin(same_status_ids)]
    
    if len(same_status_data) > 0:
        # Group by shipment_id and calculate average temp for each
        avg_per_shipment = same_status_data.groupby('shipment_id')['temperature'].mean()
        
        # Calculate the average temp for this status
        status_avg_temp = avg_per_shipment.mean()
        status_std_temp = avg_per_shipment.std()
        
        # Selected shipment's average temp
        selected_avg_temp = ts_selected['temperature'].mean()
        
        # Calculate how different the selected shipment is
        temp_diff = selected_avg_temp - status_avg_temp
        
        col1, col2, col3 = st.columns(3)
        col1.metric(f"📊 Status Average ({shipment_status})", f"{status_avg_temp:.1f}°C")
        col2.metric("📊 Your Shipment", f"{selected_avg_temp:.1f}°C")
        
        if temp_diff > 0:
            col3.metric("📊 Difference", f"+{temp_diff:.1f}°C", delta="Warmer than average", delta_color="inverse")
            st.warning(f"⚠️ This shipment is **{temp_diff:.1f}°C warmer** than the average {shipment_status} shipment.")
        else:
            col3.metric("📊 Difference", f"{temp_diff:.1f}°C", delta="Cooler than average", delta_color="normal")
            st.success(f"✅ This shipment is **{abs(temp_diff):.1f}°C cooler** than the average {shipment_status} shipment.")
        
        # Show breach rate for this status
        breach_count_same = same_status_data[same_status_data['temperature'] > 8.0]['shipment_id'].nunique()
        total_same = len(avg_per_shipment)
        if total_same > 0:
            breach_rate = (breach_count_same / total_same) * 100
            st.info(f"📊 **{breach_rate:.1f}%** of {shipment_status} shipments had at least one temperature breach (>{8}°C).")
    
    # --- Plot the Temperature Curve ---
    st.subheader("📈 Temperature Profile")
    fig = go.Figure()
    
    # Add the selected shipment
    fig.add_trace(go.Scatter(
        x=ts_selected['time_minutes'], 
        y=ts_selected['temperature'], 
        mode='lines+markers', 
        name=f'{selected_id} ({shipment_status})',
        line=dict(color='blue', width=3)
    ))
    
    # Add the average curve for the same status
    if len(same_status_data) > 0:
        # Calculate average curve
        time_points = ts_selected['time_minutes'].values
        all_curves = []
        for sid in avg_per_shipment.index[:10]:  # Limit to 10 for performance
            curve = ts_df_filtered[ts_df_filtered['shipment_id'] == sid].sort_values('time_minutes')['temperature'].values
            if len(curve) == len(time_points):
                all_curves.append(curve)
        
        if len(all_curves) > 0:
            avg_curve = np.mean(all_curves, axis=0)
            std_curve = np.std(all_curves, axis=0)
            
            fig.add_trace(go.Scatter(
                x=time_points, 
                y=avg_curve, 
                mode='lines', 
                name=f'Average ({shipment_status})',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            # Add shaded error band
            fig.add_trace(go.Scatter(
                x=time_points.tolist() + time_points.tolist()[::-1],
                y=(avg_curve + std_curve).tolist() + (avg_curve - std_curve).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='±1 Std Dev',
                showlegend=True
            ))
    
    fig.add_hline(y=8.0, line_dash="dash", line_color="green", annotation_text="⚠️ Breach Threshold (8°C)")
    fig.update_layout(
        title=f"Temperature Profile: {selected_id} (Status: {shipment_status})",
        xaxis_title="Time (minutes)",
        yaxis_title="Temperature (°C)",
        height=450,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- GPS Route Map with START/END Markers ---
    st.subheader("🗺️ GPS Route (Colored by Temperature)")
    
    # Get first and last points for markers
    start_point = ts_selected.iloc[0]
    end_point = ts_selected.iloc[-1]
    
    fig_map = px.scatter_mapbox(
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
    
    # Add start marker (Green)
    fig_map.add_scattermapbox(
        lat=[start_point['latitude']],
        lon=[start_point['longitude']],
        mode='markers',
        marker=dict(size=14, color='green'),
        text=['🟢 Start'],
        hoverinfo='text',
        name='Start'
    )
    
    # Add end marker (Red)
    fig_map.add_scattermapbox(
        lat=[end_point['latitude']],
        lon=[end_point['longitude']],
        mode='markers',
        marker=dict(size=14, color='red'),
        text=['🔴 End'],
        hoverinfo='text',
        name='End'
    )
    
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_traces(marker=dict(size=8))
    st.plotly_chart(fig_map, use_container_width=True)
    
    # --- Breach Detection ---
    breach_df = ts_selected[ts_selected['temperature'] > 8.0]
    if len(breach_df) > 0:
        first_breach = breach_df.iloc[0]
        breach_time = first_breach['time_minutes']
        st.warning(f"""
        ⚠️ **Breach Detected**: The selected shipment breached the 8°C threshold at **{breach_time} minutes**.
        - **Location**: Latitude {first_breach['latitude']:.4f}, Longitude {first_breach['longitude']:.4f}
        - **Recommendation**: Check if this location corresponds to a known traffic bottleneck or delivery stop.
        """)
    else:
        st.success("✅ No breaches detected for the selected shipment.")
