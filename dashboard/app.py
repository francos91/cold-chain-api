import streamlit as st
import pandas as pd
from supabase import create_client

# 1. Page Configuration
st.set_page_config(page_title="Logistics Pipeline Dashboard", layout="wide")
st.title("Cold Chain Logistics Pipeline Dashboard")

# 2. Secure Supabase Connection
@st.cache_resource
def init_connection():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = init_connection()

# 3. Data Fetching with Caching
@st.cache_data(ttl=600)
def get_data():
    response = supabase.table("cold_chain_tracking").select("*").execute()
    return pd.DataFrame(response.data)

# 4. Dashboard UI
df = get_data()

# Summary Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Shipments", len(df))
col2.metric("Avg Temp (°C)", round(df['temperature_celsius'].mean(), 2))
col3.metric("Delayed Shipments", len(df[df['status'] == 'delayed']))

# EDA: Visualization
st.subheader("Temperature Distribution")
st.bar_chart(df['temperature_celsius'].value_counts().sort_index())

st.subheader("Raw Logistics Data")
st.dataframe(df)
