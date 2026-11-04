from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client

# ── 1. Page Configuration ─────────────────────────────────────────────────
st.set_page_config(page_title="Logistics Dashboard", layout="wide")
st.title("Cold Chain Logistics Dashboard")


# ── 2. Constants ──────────────────────────────────────────────────────────
BREACH_THRESHOLD_C = 8.0
COST_PER_KM_ZAR = 18.0
FIXED_COST_ZAR = 200.0
DIESEL_EMISSION_FACTOR = 73.2
GRID_BEV_EMISSION_FACTOR = 156.8
SOLAR_BEV_EMISSION_FACTOR = 6.4
CARBON_TAX_RATE = 28.0
IMPACT_PER_BREACH_ZAR = 50_000.0

MODEL_FEATURES = [
    "route_enc",
    "route_risk_score",
    "is_delayed",
    "distance_km",
    "transit_time_hours",
]


# ── 3. Connection ─────────────────────────────────────────────────────────
@st.cache_resource
def init_connection():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


supabase = init_connection()


# ── 4. Data Fetch (paginated — no 1000-row cap) ───────────────────────────
def _fetch_all(table: str, page_size: int = 1000) -> pd.DataFrame:
    rows, start = [], 0
    while True:
        resp = (
            supabase.table(table)
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def get_data() -> pd.DataFrame:
    return _fetch_all("cold_chain_tracking")


@st.cache_data(ttl=600)
def get_temperature_stream() -> pd.DataFrame:
    return _fetch_all("temperature_stream")


# ── 5. Load Model & Encoder ───────────────────────────────────────────────
@st.cache_resource
def load_advanced_model():
    base = Path(__file__).parent
    model = joblib.load(base / "advanced_model.pkl")
    le_route = joblib.load(base / "le_route.pkl")
    return model, le_route


@st.cache_resource
def load_frozen_route_risk_map():
    path = Path(__file__).parent / "route_risk_map.pkl"
    return joblib.load(path) if path.exists() else None


model, le_route = load_advanced_model()
frozen_route_risk_map = load_frozen_route_risk_map()


# ── 6. Helpers ────────────────────────────────────────────────────────────
def _numeric(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def _deterministic_utilization(ids: pd.Series) -> pd.Series:
    return 70 + ids.astype(str).apply(
        lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 21
    )


def _normalise_status(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip().str.lower()
        .str.replace(r"[\s_]+", "-", regex=True)
    )


def build_route_risk_map(df: pd.DataFrame) -> dict:
    """Route-level historical breach rate. Frozen map preferred; else from full df."""
    if frozen_route_risk_map is not None:
        return frozen_route_risk_map
    route = df["origin"] + "_to_" + df["destination"]
    return (
        df.assign(
            _risk=(df["temperature_celsius"] > BREACH_THRESHOLD_C),
            _route=route,
        )
        .groupby("_route")["_risk"].mean().to_dict()
    )


def engineer_features_for_app(df: pd.DataFrame, route_risk_map: dict) -> pd.DataFrame:
    df_feat = df.copy()

    df_feat["status"] = _normalise_status(df_feat["status"])
    df_feat["route"] = df_feat["origin"] + "_to_" + df_feat["destination"]

    fallback = float(np.mean(list(route_risk_map.values()))) if route_risk_map else 0.0
    df_feat["route_risk_score"] = df_feat["route"].map(route_risk_map).fillna(fallback)

    try:
        df_feat["route_enc"] = le_route.transform(df_feat["route"])
    except ValueError:
        df_feat["route_enc"] = 0

    df_feat["is_delayed"] = (df_feat["status"] == "delayed").astype(int)
    df_feat["is_risk"] = (df_feat["temperature_celsius"] > BREACH_THRESHOLD_C).astype(int)

    df_feat["distance_km"] = _numeric(df_feat, "distance_km", 500.0)
    df_feat["weight_tonnes"] = _numeric(df_feat, "weight_tonnes", 20.0)

    transit = _numeric(df_feat, "transit_time_hours", np.nan)
    if transit.isnull().all():
        rng = np.random.default_rng(42)
        base = df_feat["distance_km"] / 65.0
        penalty = df_feat["is_delayed"] * rng.uniform(4, 12, len(df_feat))
        noise = rng.uniform(-1, 2, len(df_feat))
        transit = base + penalty + noise
    else:
        transit = transit.fillna(transit.median())
    df_feat["transit_time_hours"] = transit

    df_feat["is_ontime"] = (df_feat["status"] != "delayed").astype(int)
    df_feat["is_compliant"] = (df_feat["is_risk"] == 0).astype(int)
    df_feat["is_perfect"] = (
        (df_feat["is_ontime"] == 1) & (df_feat["is_compliant"] == 1)
    ).astype(int)

    df_feat["cost_per_shipment"] = (
        df_feat["distance_km"] * COST_PER_KM_ZAR + FIXED_COST_ZAR
    )
    df_feat["vaR_ZAR"] = df_feat["route_risk_score"] * IMPACT_PER_BREACH_ZAR
    df_feat["capacity_utilization"] = _deterministic_utilization(df_feat["shipment_id"])
    return df_feat


# ── 7. Sidebar Filtering & Cache Control ──────────────────────────────────
st.sidebar.header("Filter Options")

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

df = get_data()
if df.empty:
    st.error("No shipment data returned from Supabase.")
    st.stop()

df["status"] = _normalise_status(df["status"])
st.sidebar.write(f"Rows fetched from API: {len(df)}")

if frozen_route_risk_map is None:
    st.sidebar.warning(
        "`route_risk_map.pkl` not found — route risk is estimated from the live "
        "dataset. Ship a frozen map from training to match model semantics."
    )

selected_status = st.sidebar.multiselect(
    "Select Status",
    options=sorted(df["status"].unique()),
    default=sorted(df["status"].unique()),
)

route_risk_map = build_route_risk_map(df)
df_ready = engineer_features_for_app(df, route_risk_map)

filtered_df = df[df["status"].isin(selected_status)].copy()
filtered_df_ready = df_ready[df_ready["status"].isin(selected_status)].copy()

# Feature-contract guard
feature_names = list(getattr(model, "feature_names_in_", MODEL_FEATURES))
if feature_names != MODEL_FEATURES:
    st.sidebar.error(
        f"Model expects {feature_names}; app supplies {MODEL_FEATURES}. "
        "Predictions may be unreliable."
    )

# Pre-compute probabilities
if not filtered_df_ready.empty:
    X_pred = filtered_df_ready[MODEL_FEATURES]
    filtered_df_ready["risk_probability"] = model.predict_proba(X_pred)[:, 1]
else:
    filtered_df_ready["risk_probability"] = 0.0


# ── 8. Intermodal Corridor Reference Database ─────────────────────────────
INTERMODAL_CORRIDOR_DATA = {
    "Johannesburg_to_Durban (Natcor)": {
        "route_key": "Natcor",
        "origin": "Johannesburg",
        "destination": "Durban",
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
        "origin": "Johannesburg",
        "destination": "Cape Town",
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


# ── 9. Tabs Setup ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Raw Data",
    "Risk Intelligence",
    "Thermal Profile",
    "Intermodal Simulator",
])


# ===== TAB 1: OVERVIEW ===================================================
with tab1:
    st.subheader("SCOR-Aligned Performance Overview")
    st.caption(
        f"Active Status Filter: **{', '.join(selected_status) or '—'}** | "
        f"Total Rows in View: **{len(filtered_df_ready):,}**"
    )

    if filtered_df_ready.empty:
        st.warning("No rows match the current filter.")
    else:
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

        if "distance_km" in filtered_df_ready.columns:
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
                corridor_df,
                use_container_width=True,
                column_config={
                    "origin": "Origin",
                    "destination": "Destination",
                    "avg_transit_hours": st.column_config.NumberColumn("Avg Transit", format="%.1f hrs"),
                    "avg_distance_km": st.column_config.NumberColumn("Avg Distance", format="%.1f km"),
                    "shipment_count": st.column_config.NumberColumn("Shipments"),
                },
            )

        st.divider()

        st.subheader("Cost-to-Serve & Route Financial Exposure")
        st.markdown(
            "Evaluating transport cost efficiency and financial exposure (VaR) across"
            " major logistics corridors."
        )

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
                cost_df,
                use_container_width=True,
                column_config={
                    "origin": "Origin",
                    "destination": "Destination",
                    "avg_cost_zar": st.column_config.NumberColumn("Avg Cost", format="R%.2f"),
                    "total_shipments": st.column_config.NumberColumn("Shipments"),
                    "avg_transit_h": st.column_config.NumberColumn("Avg Transit", format="%.1f hrs"),
                },
            )

        with col_b:
            plot_data = filtered_df_ready.copy()
            sample_df = plot_data.sample(
                n=min(500, len(plot_data)), random_state=42
            )

            fig_cost = px.scatter(
                sample_df,
                x="distance_km",
                y="cost_per_shipment",
                color="status",
                title="Shipment Cost vs. Distance Physics",
                labels={
                    "distance_km": "Distance (km)",
                    "cost_per_shipment": "Cost (ZAR)",
                    "status": "Status",
                },
                color_discrete_map={
                    "delivered": "#008080",
                    "delayed": "#FF4B4B",
                    "in-transit": "#FFA500",
                },
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        st.divider()

        st.subheader("Carbon Emissions & Financial Exposure (SCOR Cost-to-Serve)")
        st.markdown(
            "Quantifying environmental impact and carbon tax liability using ISO"
            " 14083 activity-based modelling."
        )

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

        st.divider()

        st.subheader("Temperature Distribution")
        fig = px.histogram(
            filtered_df,
            x="temperature_celsius",
            nbins=20,
            color_discrete_sequence=["#008080"],
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("SCOR Framework Alignment"):
            st.markdown("""
                **Reliability (RL.1.1)** - Perfect Order Fulfillment (On-Time + No Breach)  
                **Responsiveness (RS.1.1)** - Average Transit Time (hours)  
                **Agility (AG.1.3)** - Value at Risk (route_risk_score x impact)  
                **Cost (CO.3.15)** - Cost per Shipment (distance x rand/km + fixed)  
                **Asset Management (AM.3.9)** - Capacity Utilization (simulated)  
                """)


# ===== TAB 2: RAW DATA ===================================================
with tab2:
    st.subheader("Logistics Data Table")
    st.caption(f"Displaying rows for status: **{', '.join(selected_status) or '—'}**")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "temperature_celsius": st.column_config.NumberColumn(
                "Temperature (°C)", format="%.2f"
            )
        },
    )


# ===== TAB 3: RISK INTELLIGENCE ==========================================
with tab3:
    st.subheader("Predictive Risk Intelligence")
    st.info(
        f"Active View Selections | Status Filter: `{', '.join(selected_status) or '—'}` |"
        f" Total Shipments Evaluated: `{len(filtered_df_ready):,}`"
    )

    @st.fragment
    def render_risk_fragment(data: pd.DataFrame):
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

        data = data.copy()
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

        display_cols = [
            "shipment_id",
            "origin",
            "destination",
            "temperature_celsius",
            "status",
            "risk_probability",
        ]

        def _display(df_):
            # Display as a true percentage (multiply first, then format)
            out = df_[display_cols].copy()
            out["risk_probability"] = out["risk_probability"] * 100
            st.dataframe(
                out,
                use_container_width=True,
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

        if len(top_5_df) > 0:
            _display(top_5_df)

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
            _display(high_risk_df)
            st.caption(
                "Showing "
                f"{len(high_risk_df)} active high-risk shipments at the"
                f" **{threshold_pct}%** threshold."
            )
        else:
            st.success(
                f"No active high-risk shipments at the **{threshold_pct}%** threshold!"
            )

    if not filtered_df_ready.empty:
        render_risk_fragment(filtered_df_ready)

    st.markdown("### High-Risk Routes Analysis")

    if not filtered_df_ready.empty:
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
        st.plotly_chart(fig_risk, use_container_width=True)

    st.info(
        "The model analysis indicates that spatial routing characteristics"
        " (route_risk_score) are the primary drivers of cold chain breaches,"
        " providing a data-driven basis for infrastructure investment in"
        " high-risk corridors."
    )


# ===== TAB 4: THERMAL PROFILE ============================================
with tab4:
    st.subheader("Thermal Profile Analysis")
    st.markdown(
        "View the temperature curve and GPS route for a shipment, compared to the"
        " average for its status."
    )
    st.caption(f"Active Status Filter: **{', '.join(selected_status) or '—'}**")

    def _render_thermal():
        if not selected_status:
            st.info("No status selected in the sidebar. Please select at least one status.")
            return
        if filtered_df.empty:
            st.info(
                "No shipments found with status(es):"
                f" {', '.join(selected_status)}"
            )
            return

        with st.spinner("Loading thermal profile data..."):
            ts_df = get_temperature_stream()

        if len(ts_df) == 0:
            st.warning(
                "No time-series data found. Please run the temperature_stream generator"
                " first."
            )
            return

        # Derive IDs from the already-paginated master table (no extra capped query)
        filtered_shipment_ids = filtered_df["shipment_id"].tolist()
        ts_df_filtered = ts_df[ts_df["shipment_id"].isin(filtered_shipment_ids)]

        if len(ts_df_filtered) == 0:
            st.info(
                "No time-series data found for shipments with status(es):"
                f" {', '.join(selected_status)}"
            )
            return

        shipment_ids = ts_df_filtered["shipment_id"].unique()
        selected_id = st.selectbox("Select Shipment to Analyze", shipment_ids, index=0)

        ts_selected = (
            ts_df_filtered[ts_df_filtered["shipment_id"] == selected_id]
            .sort_values("time_minutes")
        )

        if len(ts_selected) == 0:
            st.warning("No temperature data found for this shipment.")
            return

        row = filtered_df[filtered_df["shipment_id"] == selected_id]
        if row.empty:
            st.warning("Shipment not found in the main table.")
            return
        shipment_status = row.iloc[0]["status"]

        st.subheader("Quick Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Shipment", selected_id)
        col2.metric("Status", shipment_status.capitalize())
        col3.metric("Avg Temp", f"{ts_selected['temperature'].mean():.1f}°C")

        st.subheader("Comparison to Average")

        same_status_ids = set(
            filtered_df.loc[
                filtered_df["status"] == shipment_status, "shipment_id"
            ]
        )
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
                col3.metric(
                    "Difference",
                    f"+{temp_diff:.1f}°C",
                    delta="Warmer than average",
                    delta_color="inverse",
                )
                st.warning(
                    "This shipment is **"
                    f"{temp_diff:.1f}°C warmer** than the average {shipment_status}"
                    " shipment."
                )
            else:
                col3.metric(
                    "Difference",
                    f"{temp_diff:.1f}°C",
                    delta="Cooler than average",
                    delta_color="normal",
                )
                st.success(
                    "This shipment is **"
                    f"{abs(temp_diff):.1f}°C cooler** than the average {shipment_status}"
                    " shipment."
                )

            breach_count_same = same_status_data[
                same_status_data["temperature"] > BREACH_THRESHOLD_C
            ]["shipment_id"].nunique()
            total_same = len(avg_per_shipment)
            if total_same > 0:
                breach_rate = (breach_count_same / total_same) * 100
                st.info(
                    "**"
                    f"{breach_rate:.1f}%** of {shipment_status} shipments had at least one"
                    f" temperature breach (>{BREACH_THRESHOLD_C:.0f}°C)."
                )

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
            y=BREACH_THRESHOLD_C,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Breach Threshold ({BREACH_THRESHOLD_C:.0f}°C)",
        )
        fig.update_layout(
            title=f"Temperature Profile: {selected_id} (Status: {shipment_status})",
            xaxis_title="Time (minutes)",
            yaxis_title="Temperature (°C)",
            height=450,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

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

        # Apply size to the main trace first, then add start/end markers
        fig_map.update_traces(marker=dict(size=8))

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
        st.plotly_chart(fig_map, use_container_width=True)

        breach_df = ts_selected[ts_selected["temperature"] > BREACH_THRESHOLD_C]
        if len(breach_df) > 0:
            first_breach = breach_df.iloc[0]
            breach_time = first_breach["time_minutes"]
            st.warning(
                f"**Breach Detected**: The selected shipment breached the"
                f" {BREACH_THRESHOLD_C:.0f}°C threshold "
                f"at **{breach_time} minutes**.\n"
                f"- **Location**: Latitude {first_breach['latitude']:.4f}, Longitude {first_breach['longitude']:.4f}\n"
                f"- **Recommendation**: Check if this location corresponds to a known traffic bottleneck or delivery stop."
            )
        else:
            st.success("No breaches detected for the selected shipment.")

    # Lazy-load gate: 66k-row telemetry fetch runs only on button click
    if "thermal_loaded" not in st.session_state:
        st.session_state.thermal_loaded = False

    if st.button("Load thermal telemetry (~66,000 rows)") or st.session_state.thermal_loaded:
        st.session_state.thermal_loaded = True
        _render_thermal()
    else:
        st.info(
            "Click above to load thermal profiles. First fetch takes 30–60s "
            "on cold start; then cached for 10 min."
        )


# ===== TAB 5: INTERMODAL WHAT-IF SIMULATOR ===============================
with tab5:
    st.subheader("Intermodal Mode-Switching")
    st.markdown(
        "Evaluate the carbon, tax, and cost-to-serve trade-offs of shifting"
        " long-haul refrigerated freight from road corridors (N3/N1) to Transnet"
        " Freight Rail (TFR)."
    )

    @st.fragment
    def render_intermodal_fragment(data: pd.DataFrame, corridor_data: dict):
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
        s = shift_percentage / 100.0

        # Corridor cohort (falls back to full view if no shipments match)
        corridor_df = data[
            (data["origin"].astype(str).str.strip().str.lower()
             == str(params["origin"]).strip().lower())
            & (data["destination"].astype(str).str.strip().str.lower()
               == str(params["destination"]).strip().lower())
        ]
        used_full_network = False
        if corridor_df.empty:
            corridor_df = data
            used_full_network = True

        n = len(corridor_df)
        avg_w = corridor_df["weight_tonnes"].mean() if n else 20.0

        # Cost model — consistent with Tab 1 (R18/km + R200/shipment road, rail R/km + fixed)
        road_per = params["road_distance_km"] * COST_PER_KM_ZAR + FIXED_COST_ZAR
        rail_per = (
            params["rail_distance_km"] * params["rail_cost_per_km"] + FIXED_COST_ZAR
        )
        terminal = params["terminal_handling_zar"] if s > 0 else 0.0

        baseline_cost = n * road_per
        simulated_cost = (1 - s) * n * road_per + s * n * rail_per + terminal

        # Emissions — per-shipment tkm method (same physics as Tab 1)
        def _corridor_emissions(dist_km: float, ef: float) -> float:
            return n * avg_w * dist_km * ef / 1_000_000

        baseline_emissions_tonnes = _corridor_emissions(
            params["road_distance_km"], params["road_emission_factor"]
        )
        new_emissions_tonnes = (
            (1 - s) * _corridor_emissions(
                params["road_distance_km"], params["road_emission_factor"]
            )
            + s * _corridor_emissions(
                params["rail_distance_km"], params["rail_emission_factor"]
            )
        )
        carbon_savings_tonnes = baseline_emissions_tonnes - new_emissions_tonnes
        carbon_tax_savings_zar = carbon_savings_tonnes * CARBON_TAX_RATE

        # Transit
        avg_road_hours = params["road_distance_km"] / params["avg_road_speed_kmh"]
        avg_rail_hours = (
            params["rail_distance_km"] / params["avg_rail_speed_kmh"]
        ) + 18.0
        transit_time_delta_hrs = s * (avg_rail_hours - avg_road_hours)

        # 5 metrics: original 4 + cost delta
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            "Carbon Emissions Delta",
            f"-{carbon_savings_tonnes:.2f} tCO2",
            delta=f"{shift_percentage}% Rail Shift",
        )
        m2.metric(
            "Carbon Tax Liability Saved",
            f"R {carbon_tax_savings_zar:,.2f}",
            delta="Financial Benefit",
            delta_color="normal",
        )
        m3.metric(
            "Modal Distance Factor",
            f"{params['rail_distance_km']} km",
            delta=f"Road: {params['road_distance_km']} km",
            delta_color="off",
        )
        m4.metric(
            "Transit Time Impact",
            f"+{transit_time_delta_hrs:.1f} hrs",
            delta="Terminal Delay Penalty",
            delta_color="inverse",
        )
        m5.metric(
            "Cost Δ vs 100% Road",
            f"R {simulated_cost - baseline_cost:,.0f}",
            delta=(
                f"{(simulated_cost - baseline_cost) / baseline_cost * 100:+.1f}%"
                if baseline_cost else "—"
            ),
            delta_color="inverse" if simulated_cost > baseline_cost else "normal",
        )

        if used_full_network:
            st.caption(
                "⚠️ No live shipments matched this corridor — using the full "
                "filtered dataset as the cohort."
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
                simulated_cost,
                avg_road_hours + transit_time_delta_hrs,
            ],
        })
        
        st.dataframe(
            comparison_df.set_index("Metric"),
            use_container_width=True,
            column_config={
                "Baseline (100% Road)": st.column_config.NumberColumn(format="%.2f"),
                f"Simulated ({shift_percentage}% Rail)": st.column_config.NumberColumn(format="%.2f"),
            }
        )

    render_intermodal_fragment(filtered_df_ready, INTERMODAL_CORRIDOR_DATA)
