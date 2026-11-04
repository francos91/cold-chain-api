import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Diagnostic", layout="wide")
st.title("Cold Chain — Diagnostic")
st.write("If you see this, the app boots.")

# 1. Secrets
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    st.success(f"✅ Secrets present (URL len {len(url)}, key len {len(key)})")
except Exception as e:
    st.error(f"❌ SECRETS MISSING: {type(e).__name__}: {e}")
    st.stop()

# 2. Imports
try:
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import joblib
    st.success(f"✅ Imports OK — pandas {pd.__version__}, numpy {np.__version__}")
except Exception as e:
    st.error(f"❌ Import failed: {e}")
    st.stop()

# 3. Model
try:
    m = joblib.load(Path(__file__).parent / "advanced_model.pkl")
    st.success(f"✅ Model loaded — {type(m).__name__}")
except Exception as e:
    st.error(f"❌ Model load failed: {type(e).__name__}: {e}")
    st.stop()

# 4. Supabase — one call
try:
    from supabase import create_client
    c = create_client(url, key)
    r = c.table("cold_chain_tracking").select("id").limit(1).execute()
    st.success(f"✅ Supabase OK — {len(r.data)} row(s)")
except Exception as e:
    st.error(f"❌ Supabase failed: {type(e).__name__}: {e}")
    st.stop()

# 5. Route risk map
try:
    rrm = joblib.load(Path(__file__).parent / "route_risk_map.pkl")
    st.success(f"✅ route_risk_map.pkl loaded — {len(rrm)} routes")
except FileNotFoundError:
    st.warning("⚠️ route_risk_map.pkl missing (not fatal — fallback exists)")
except Exception as e:
    st.error(f"❌ route_risk_map failed: {e}")

st.write("All checks done.")
