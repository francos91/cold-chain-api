import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
import joblib
from pathlib import Path

# 1. Page Configuration
st.set_page_config(page_title="Logistics Dashboard", layout="wide")
st.title("Cold Chain Logistics Dashboard")

# 2. Connection
@st.cache_resource
def init_connection():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = init_connection()

@st.cache_data(ttl=600)
def get_data():
    # .limit(5000) ensures we pull the full dataset without default caps
    response = supabase.table("cold_chain_tracking").select("*").limit(5000).execute()
    return pd.DataFrame(response.data)

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
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📋 Raw Data", "🚀 Risk Intelligence"])

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
# UPDATED TAB 3: WITH RISK THRESHOLD SLIDER
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
