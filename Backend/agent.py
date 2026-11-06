import os
import json
import logging
from typing import List, Dict
import requests
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
from google.genai import types

logger = logging.getLogger("coldchain-agent")
router = APIRouter(prefix="/api/v1/agent", tags=["AI Operations Copilot"])

# 1. Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set in environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Load containerized ML assets
try:
    rf_model = joblib.load("advanced_model.pkl")
    route_encoder = joblib.load("le_route.pkl")
    logger.info("Agent successfully loaded ML models.")
except Exception as e:
    logger.warning(f"Agent failed to load ML models: {e}")
    rf_model, route_encoder = None, None


# --- 2. EXECUTABLE TOOLS FOR GEMINI ---

def get_active_fleet_telemetry(only_breaches: bool = False) -> str:
    """
    Fetches live telemetry for active (non-delivered) shipments from Supabase.
    Set only_breaches=True to filter exclusively for shipments above the 8.0°C safety threshold.
    """
    try:
        query = (
            supabase.table("cold_chain_tracking")
            .select("shipment_id,origin,destination,temperature_celsius,status,distance_km,transit_time_hours,weight_tonnes")
            .not_.eq("status", "delivered")
        )
        if only_breaches:
            query = query.gt("temperature_celsius", 8.0)
        res = query.limit(50).execute()
        return json.dumps(res.data or [])
    except Exception as e:
        return json.dumps({"error": f"Database query failed: {str(e)}"})


def predict_temperature_breach_risk(origin: str, destination: str, is_delayed: bool, transit_time_hours: float, distance_km: float = 600.0) -> dict:
    """
    Runs the containerized 5-Feature Random Forest classifier to predict the continuous probability of a Cold Chain breach (>8°C).
    Use this for what-if scenarios, dispatch scheduling, and custom delay inquiries.
    """
    if rf_model is None or route_encoder is None:
        return {"error": "ML models not available in backend container."}

    route_key = f"{origin}_to_{destination}"
    try:
        route_enc = int(route_encoder.transform([route_key])[0])
    except Exception:
        route_enc = 0

    route_risk_proxy = 0.22 if is_delayed else 0.08
    delayed_flag = 1 if is_delayed else 0

    features = pd.DataFrame([{
        "route_enc": route_enc,
        "route_risk_score": route_risk_proxy,
        "is_delayed": delayed_flag,
        "distance_km": float(distance_km),
        "transit_time_hours": float(transit_time_hours)
    }])

    prob = float(rf_model.predict_proba(features)[0, 1])
    return {
        "route": route_key,
        "breach_probability_pct": round(prob * 100, 2),
        "risk_classification": "CRITICAL RISK" if prob > 0.40 else "MODERATE RISK" if prob > 0.15 else "LOW RISK",
        "parameters": {
            "is_delayed": is_delayed,
            "transit_time_hours": transit_time_hours,
            "distance_km": distance_km
        }
    }


def get_corridor_live_weather(location_name: str) -> dict:
    """
    Fetches real-time ambient weather (temperature, humidity, pressure) from Open-Meteo.
    Supported key South African waypoints: 'Johannesburg', 'Durban', 'Cape Town', 'Beaufort West', 'Bloemfontein', 'Harrismith'.
    """
    COORDINATES = {
        "johannesburg": (-26.2041, 28.0473),
        "durban": (-29.8587, 31.0218),
        "cape town": (-33.9249, 18.4241),
        "beaufort west": (-32.3567, 22.5830),
        "bloemfontein": (-29.0852, 26.1596),
        "harrismith": (-28.2728, 29.1297),
    }
    lat, lon = COORDINATES.get(location_name.strip().lower(), (-26.2041, 28.0473))
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        current = r.json().get("current", {})
        return {
            "waypoint": location_name,
            "ambient_temp_celsius": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "surface_pressure_hpa": current.get("surface_pressure"),
            "highveld_thermal_penalty": "21% altitude cooling efficiency loss applied" if current.get("surface_pressure", 1013) < 850 else "Nominal cooling"
        }
    except Exception as e:
        return {"error": f"Weather fetch failed: {str(e)}"}


def simulate_modal_shift_economics(corridor_key: str, shift_percentage: float, total_tkm: float = 600000.0) -> dict:
    """
    Calculates carbon emission reductions and South African Carbon Tax savings (R28.00/tonne) for shifting road freight to rail.
    Corridors: 'Johannesburg_to_Durban (Natcor)' or 'Johannesburg_to_Cape_Town (Capecor)'.
    """
    road_emission_factor = 73.2
    rail_emission_factor = 12.2
    effective_tax_rate_zar = 28.0

    baseline_emissions = (total_tkm * road_emission_factor) / 1_000_000
    rail_tkm = total_tkm * (shift_percentage / 100.0)
    remaining_road_tkm = total_tkm * (1 - (shift_percentage / 100.0))

    new_emissions = ((remaining_road_tkm * road_emission_factor) + (rail_tkm * rail_emission_factor)) / 1_000_000
    carbon_saved_tco2 = baseline_emissions - new_emissions
    tax_savings_zar = carbon_saved_tco2 * effective_tax_rate_zar

    return {
        "corridor": corridor_key,
        "modal_shift_pct": shift_percentage,
        "carbon_saved_tco2": round(carbon_saved_tco2, 2),
        "carbon_tax_savings_zar": round(tax_savings_zar, 2),
        "baseline_emissions_tco2": round(baseline_emissions, 2),
        "new_emissions_tco2": round(new_emissions, 2)
    }


SYSTEM_PROMPT = """You are the AI Control Tower Copilot for a South African Cold Chain Logistics Network.
You have native access to executable tools:
- `get_active_fleet_telemetry`: queries live active shipments in transit.
- `predict_temperature_breach_risk`: runs the container's 5-feature Random Forest model.
- `get_corridor_live_weather`: calls the live Open-Meteo API for ambient temperatures along key waypoints.
- `simulate_modal_shift_economics`: evaluates ISO 14083 carbon accounting and R28/tCO2 tax liability.

Operational Rules:
1. Perishable excursion threshold is strictly 8.0°C. Anything higher is an active CRITICAL BREACH.
2. If the user asks a predictive or 'what-if' question, invoke `predict_temperature_breach_risk`.
3. If the user asks about weather, ambient heat, or corridor conditions, invoke `get_corridor_live_weather`.
4. Always provide quantitative figures, clean bullet points, and actionable next steps.
"""


# --- 3. FASTAPI ROUTE ---

class AgentQueryRequest(BaseModel):
    user_query: str

class AgentQueryResponse(BaseModel):
    response: str


@router.post("/query", response_model=AgentQueryResponse)
def query_agent(payload: AgentQueryRequest):
    if not ai_client:
        raise HTTPException(
            status_code=503,
            detail="Gemini API Client not initialized. Set GEMINI_API_KEY in Render environment."
        )

    try:
        # Pass Python functions directly into tools list for automatic function calling
        response = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=payload.user_query,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[
                    get_active_fleet_telemetry,
                    predict_temperature_breach_risk,
                    get_corridor_live_weather,
                    simulate_modal_shift_economics
                ],
                temperature=0.2,
            ),
        )
        return AgentQueryResponse(response=response.text)

    except Exception as e:
        logger.exception("Agent call failed")
        raise HTTPException(status_code=500, detail=str(e))
