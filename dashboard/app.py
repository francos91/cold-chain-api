import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client

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
    response = supabase.table("cold_chain_tracking").select("*").execute()
    return pd.DataFrame(response.data)

df = get_data()

# 3. Sidebar Filtering
st.sidebar.header("Filter Options")
selected_status = st.sidebar.multiselect("Select Status", options=df['status'].unique(), default=df['status'].unique())
filtered_df = df[df['status'].isin(selected_status)]

# 4. Tabs and Professional Layout
tab1, tab2 = st.tabs(["📊 Overview", "📋 Raw Data"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Shipments", len(filtered_df))
    
    # Calculate Average and color-code logic
    avg_temp = round(filtered_df['temperature_celsius'].mean(), 2)
    col2.metric("Avg Temp (°C)", avg_temp)
    
    col3.metric("Delayed Shipments", len(filtered_df[filtered_df['status'] == 'delayed']))
    
    st.subheader("Temperature Distribution")
    # Professional Plotly Histogram
    fig = px.histogram(filtered_df, x="temperature_celsius", 
                       nbins=20, 
                       color_discrete_sequence=['#008080'])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Logistics Data Table")
    # Highlight dangerous temperatures (> 8°C) in red
    def color_risk(val):
        color = 'red' if val > 8 else 'green'
        return f'color: {color}'
    
    st.dataframe(filtered_df.style.applymap(color_risk, subset=['temperature_celsius']), 
                 use_container_width=True)
