import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Intermodal Simulator", layout="wide")
st.title("Intermodal Mode-Switching Simulator")
st.markdown("Evaluate the carbon, tax, and cost-to-serve trade-offs of shifting long-haul refrigerated freight from road to rail. All heavy physics and financial calculations are executed on the Render microservice.")

API_DASHBOARD = "https://cold-chain-api-docker.onrender.com/api/dashboard"
API_SIMULATE = "https://cold-chain-api-docker.onrender.com/api/simulate-intermodal"

@st.cache_data(ttl=600)
def get_base_data():
    try:
        res = requests.get(API_DASHBOARD)
        res.raise_for_status()
        return pd.DataFrame(res.json()["features"])
    except:
        return pd.DataFrame()

df_feat = get_base_data()

corridors = ["Johannesburg_to_Durban (Natcor)", "Johannesburg_to_Cape_Town (Capecor)"]

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    selected_corridor = st.selectbox("Select Strategic Corridor", options=corridors)
with col_ctrl2:
    shift_percentage = st.slider("Percentage of Freight Shifted from Road to Rail (%)", 0, 100, 30, step=5)

# 1. Calculate dynamic inputs to send to the API based on current data
if not df_feat.empty and "distance_km" in df_feat.columns and "weight_tonnes" in df_feat.columns:
    total_tkm = float((df_feat["distance_km"] * df_feat["weight_tonnes"]).sum())
    total_shipments = int(len(df_feat))
else:
    total_tkm = 580000.0
    total_shipments = 45

# 2. Package the payload for the Render API
payload = {
    "corridor_key": selected_corridor,
    "shift_percentage": float(shift_percentage),
    "total_tkm": total_tkm,
    "total_shipments": total_shipments
}

# 3. Call the Render microservice
try:
    res = requests.post(API_SIMULATE, json=payload)
    res.raise_for_status()
    results = res.json()
    
    st.divider()
    
    # 4. Display Results calculated by the backend
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Carbon Emissions Delta", f"-{results['carbon_savings_tonnes']:.2f} tCO2", delta=f"{shift_percentage}% Rail Shift")
    col_m2.metric("Carbon Tax Saved", f"R {results['carbon_tax_savings_zar']:,.2f}", delta="Financial Benefit", delta_color="normal")
    col_m3.metric("Transit Time Impact", f"+{results['transit_time_delta_hrs']:.1f} hrs", delta="Terminal Delay Penalty", delta_color="inverse")
    
    # Cost comparison logic
    cost_diff = results['new_cost'] - results['baseline_cost']
    cost_color = "normal" if cost_diff < 0 else "inverse"
    col_m4.metric("New Fleet Cost", f"R {results['new_cost']:,.0f}", delta=f"{cost_diff:+,.0f} ZAR", delta_color=cost_color)

    st.markdown("### Modal Performance Comparison")
    comparison_df = pd.DataFrame({
        "Metric": ["Carbon Footprint (tCO2e)", "Estimated Cost Index (ZAR)", "Average Transit Time (Hours)"],
        "Baseline (100% Road)": [results['baseline_emissions_tonnes'], results['baseline_cost'], results['avg_road_hours']],
        f"Simulated ({shift_percentage}% Rail)": [results['new_emissions_tonnes'], results['new_cost'], results['new_transit_hours']]
    })
    
    st.dataframe(comparison_df.set_index("Metric"), width="stretch")
    
except Exception as e:
    st.error(f"Failed to run simulation via API: {e}")
