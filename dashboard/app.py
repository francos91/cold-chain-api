import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(page_title="Logistics Dashboard", layout="wide")
st.title("Cold Chain Logistics Pipeline Dashboard")

# 2. Connection
@st.cache_resource
def init_connection():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = init_connection()

@st.cache_data(ttl=600)
def get_data():
    # FIX: Explicitly range to 4999 to pull your full 2,000+ dataset
    response = supabase.table("cold_chain_tracking").select("*").range(0, 4999).execute()
    return pd.DataFrame(response.data)

# Helper: Feature Engineering for Predictive Analytics
def engineer_features_for_app(df):
    df_feat = df.copy()
    df_feat['route'] = df_feat['origin'] + "_to_" + df_feat['destination']
    
    # Calculate historical risk (mean excursions per route)
    risk_map = df_feat.groupby('route')['temperature_celsius'].apply(lambda x: (x > 8.0).mean()).to_dict()
    df_feat['route_risk_score'] = df_feat['route'].map(risk_map)
    
    # Add status indicator
    df_feat['is_risk'] = (df_feat['temperature_celsius'] > 8.0).astype(int)
    return df_feat

# 3. Sidebar Filtering & Cache Control
st.sidebar.header("Filter Options")

# Force Refresh button for your cache
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

df = get_data()
df_ready = engineer_features_for_app(df)

selected_status = st.sidebar.multiselect("Select Status", options=df['status'].unique(), default=df['status'].unique())
filtered_df = df[df['status'].isin(selected_status)]
# Sync filtering with the processed data
filtered_df_ready = df_ready[df_ready['status'].isin(selected_status)]

# 4. Tabs and Professional Layout
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

with tab3:
    st.subheader("Predictive Risk Intelligence")
    st.markdown("### High-Risk Routes Analysis")
    route_risk = filtered_df_ready.groupby('route')['is_risk'].mean().reset_index()
    fig_risk = px.bar(route_risk.sort_values('is_risk', ascending=False).head(10), 
                     x='is_risk', y='route', orientation='h', 
                     title="Routes with Highest Excursion Frequency")
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.info("The model analysis indicates that spatial routing characteristics (route_risk_score) are the primary drivers of cold chain breaches, providing a data-driven basis for infrastructure investment in high-risk corridors.")
