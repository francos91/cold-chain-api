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
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Shipments", len(filtered_df))
    col2.metric("Avg Temp (°C)", round(filtered_df['temperature_celsius'].mean(), 2))
    col3.metric("Delayed Shipments", len(filtered_df[filtered_df['status'] == 'delayed']))
    
    st.subheader("Temperature Distribution")
    fig = px.histogram(filtered_df, x="temperature_celsius", nbins=20, color_discrete_sequence=['#008080'])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Logistics Data Table")
    # Restored your original professional column configuration
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
    ) / 100.0  # Convert to decimal (e.g., 50 -> 0.50)
    
    st.caption(f"⚙️ Currently using a **{threshold*100:.0f}%** threshold. Shipments with a probability above this are flagged as high-risk.")
    
    # --- Prepare features for the model ---
    X_pred = filtered_df_ready[['route_enc', 'route_risk_score', 'is_delayed']]
    
    # --- Get probabilities (not just binary predictions) ---
    filtered_df_ready['risk_probability'] = model.predict_proba(X_pred)[:, 1]
    
    # --- Apply the dynamic threshold to get binary predictions ---
    filtered_df_ready['predicted_risk'] = (filtered_df_ready['risk_probability'] >= threshold).astype(int)
    
    # --- Display AI-powered metrics ---
    col1, col2 = st.columns(2)
    col1.metric("🚨 Predicted High-Risk Shipments", filtered_df_ready['predicted_risk'].sum())
    col2.metric("🎯 Avg Risk Probability", f"{filtered_df_ready['risk_probability'].mean():.1%}")
    
    st.divider()

    # --- 🚨 Top 5 Highest Risk Shipments (Leaderboard) ---
    st.markdown("### 🚨 Top 5 Highest Risk Shipments")
    
    # Filter to show only predicted high-risk shipments that are NOT already delivered
    high_risk_df = filtered_df_ready[
        (filtered_df_ready['predicted_risk'] == 1) & 
        (filtered_df_ready['status'] != 'delivered')
    ]
    
    # Sort by risk probability (highest first) and take top 5
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
        
        # Show the highest risk probability as a bold metric
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
# TAB 4: THERMAL PROFILE (SIMPLIFIED - Single Shipment vs Average)
# =============================================
with tab4:
    st.subheader("📈 Thermal Profile Analysis")
    st.markdown("Compare a shipment's temperature curve against the average pattern for similar shipments.")
    
    # Fetch time-series data
    with st.spinner("Loading thermal profile data..."):
        ts_response = supabase.table("temperature_stream").select("*").execute()
        ts_df = pd.DataFrame(ts_response.data)
    
    if len(ts_df) == 0:
        st.warning("⚠️ No time-series data found. Please run the temperature_stream generator first.")
        st.stop()
    
    # Get unique shipment IDs
    shipment_ids = ts_df['shipment_id'].unique()
    
    # --- Select a Single Shipment ---
    selected_id = st.selectbox("🔵 Select Shipment to Analyze", shipment_ids, index=0)
    
    # Fetch the selected shipment's curve
    ts_selected = ts_df[ts_df['shipment_id'] == selected_id].sort_values('time_minutes')
    
    if len(ts_selected) == 0:
        st.warning("⚠️ No temperature data found for this shipment.")
        st.stop()
    
    # Get the status of the selected shipment from the parent table
    status_resp = supabase.table("cold_chain_tracking").select("status").eq("shipment_id", selected_id).execute()
    if len(status_resp.data) == 0:
        st.warning("⚠️ Shipment not found in the main table.")
        st.stop()
    shipment_status = status_resp.data[0]['status']
    
    # --- Build the "Average Curve" for this status ---
    # Get all shipments with the same status from the parent table FIRST
    status_resp_all = supabase.table("cold_chain_tracking").select("shipment_id").eq("status", shipment_status).execute()
    same_status_ids = [row['shipment_id'] for row in status_resp_all.data]
    
    # Now filter the time-series data to only include these shipments
    same_status_ts = ts_df[ts_df['shipment_id'].isin(same_status_ids)]
    
    if len(same_status_ids) < 2:
        st.info("ℹ️ Not enough shipments with this status to build an average. Showing the selected curve only.")
        avg_curve = None
    else:
        # Calculate the average temperature at each time point
        all_curves = []
        for sid in same_status_ids[:50]:  # Limit to 50 to avoid performance issues
            curve = ts_df[ts_df['shipment_id'] == sid].sort_values('time_minutes')['temperature'].values
            if len(curve) == 12:  # Only use complete curves
                all_curves.append(curve)
        
        if len(all_curves) > 0:
            avg_curve = np.mean(all_curves, axis=0)
            std_curve = np.std(all_curves, axis=0)
        else:
            avg_curve = None
    
    # --- Display Metrics ---
    st.subheader("📊 Quick Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Shipment", selected_id)
    col2.metric("📌 Status", shipment_status.capitalize())
    col3.metric("🌡️ Avg Temp", f"{ts_selected['temperature'].mean():.1f}°C")
    
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
    
    # Add the average curve (if available)
    if avg_curve is not None:
        times = ts_selected['time_minutes'].values
        fig.add_trace(go.Scatter(
            x=times, 
            y=avg_curve, 
            mode='lines', 
            name=f'Average ({shipment_status})',
            line=dict(color='orange', width=2, dash='dash')
        ))
        # Add shaded error band
        fig.add_trace(go.Scatter(
            x=times.tolist() + times.tolist()[::-1],
            y=(avg_curve + std_curve).tolist() + (avg_curve - std_curve).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='±1 Std Dev',
            showlegend=True
        ))
        
        # Calculate how similar the selected curve is to the average
        try:
            distance, _ = fastdtw(ts_selected['temperature'].tolist(), avg_curve.tolist(), dist=euclidean)
            # Normalize similarity (lower distance = more similar)
            max_possible = 30  # Max possible DTW distance for this data
            similarity = max(0, min(100, 100 - (distance / max_possible * 100)))
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Similarity to Average", f"{similarity:.0f}%")
            if similarity > 80:
                st.success(f"✅ This shipment follows the typical pattern for {shipment_status} shipments.")
            elif similarity > 60:
                st.info(f"ℹ️ This shipment is moderately different from typical {shipment_status} shipments.")
            else:
                st.warning(f"⚠️ This shipment is significantly different from typical {shipment_status} shipments. Investigate further.")
        except:
            st.info("ℹ️ Unable to compute similarity metric.")
    
    fig.add_hline(y=8.0, line_dash="dash", line_color="green", annotation_text="⚠️ Breach Threshold (8°C)")
    fig.update_layout(
        title=f"Temperature Profile: {selected_id} (Status: {shipment_status})",
        xaxis_title="Time (minutes)",
        yaxis_title="Temperature (°C)",
        height=450,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- GPS Route Map ---
    st.subheader("🗺️ GPS Route (Colored by Temperature)")
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
        height=400
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_traces(marker=dict(size=8))
    st.plotly_chart(fig_map, use_container_width=True)
    
    # --- Breach Detection (FIXED) ---
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
