import os
import json
import logging
from typing import List, Dict, Any
import requests
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from groq import AsyncGroq

logger = logging.getLogger("coldchain-agent")
router = APIRouter(prefix="/api/v1/agent", tags=["AI Operations Copilot"])

# 1. Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set in environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Load containerized ML assets
try:
    rf_model = joblib.load("advanced_model.pkl")
    route_encoder = joblib.load("le_route.pkl")
    logger.info("Agent successfully loaded ML models.")
except Exception as e:
    logger.warning(f"Agent failed to load ML models: {e}")
    rf_model, route_encoder = None, None


# --- 2. EXECUTABLE TOOLS ---

def get_active_fleet_telemetry(only_breaches: bool = False) -> str:
    """Fetches live telemetry for active shipments from Supabase."""
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


def predict_temperature_breach_risk(origin: str, destination: str, is_delayed: bool, transit_time_hours: float, distance_km: float = 600.0) -> str:
    """Runs the containerized Random Forest model to predict continuous probability of a temperature breach (>8°C)."""
    if rf_model is None or route_encoder is None:
        return json.dumps({"error": "ML models not available in backend container."})

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
    return json.dumps({
        "route": route_key,
        "breach_probability_pct": round(prob * 100, 2),
        "risk_classification": "CRITICAL RISK" if prob > 0.40 else "MODERATE RISK" if prob > 0.15 else "LOW RISK",
        "parameters": {
            "is_delayed": is_delayed,
            "transit_time_hours": transit_time_hours,
            "distance_km": distance_km
        }
    })


def get_corridor_live_weather(location_name: str) -> str:
    """Fetches real-time ambient weather from Open-Meteo."""
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
        return json.dumps({
            "waypoint": location_name,
            "ambient_temp_celsius": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "surface_pressure_hpa": current.get("surface_pressure"),
            "highveld_thermal_penalty": "21% altitude cooling efficiency loss applied" if current.get("surface_pressure", 1013) < 850 else "Nominal cooling"
        })
    except Exception as e:
        return json.dumps({"error": f"Weather fetch failed: {str(e)}"})


def simulate_modal_shift_economics(corridor_key: str, shift_percentage: float, total_tkm: float = 600000.0) -> str:
    """Calculates carbon reductions and South African Carbon Tax savings (R28.00/tonne)."""
    road_emission_factor = 73.2
    rail_emission_factor = 12.2
    effective_tax_rate_zar = 28.0

    baseline_emissions = (total_tkm * road_emission_factor) / 1_000_000
    rail_tkm = total_tkm * (shift_percentage / 100.0)
    remaining_road_tkm = total_tkm * (1 - (shift_percentage / 100.0))

    new_emissions = ((remaining_road_tkm * road_emission_factor) + (rail_tkm * rail_emission_factor)) / 1_000_000
    carbon_saved_tco2 = baseline_emissions - new_emissions
    tax_savings_zar = carbon_saved_tco2 * effective_tax_rate_zar

    return json.dumps({
        "corridor": corridor_key,
        "modal_shift_pct": shift_percentage,
        "carbon_saved_tco2": round(carbon_saved_tco2, 2),
        "carbon_tax_savings_zar": round(tax_savings_zar, 2),
        "baseline_emissions_tco2": round(baseline_emissions, 2),
        "new_emissions_tco2": round(new_emissions, 2)
    })


AVAILABLE_TOOLS = {
    "get_active_fleet_telemetry": get_active_fleet_telemetry,
    "predict_temperature_breach_risk": predict_temperature_breach_risk,
    "get_corridor_live_weather": get_corridor_live_weather,
    "simulate_modal_shift_economics": simulate_modal_shift_economics,
}

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_active_fleet_telemetry",
            "description": "Fetches live telemetry for active (non-delivered) shipments from Supabase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "only_breaches": {
                        "type": "boolean",
                        "description": "Set to True to filter exclusively for shipments above 8.0°C."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_temperature_breach_risk",
            "description": "Runs the containerized Random Forest model to predict continuous probability of a temperature breach (>8°C).",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "is_delayed": {"type": "boolean"},
                    "transit_time_hours": {"type": "number"},
                    "distance_km": {"type": "number"}
                },
                "required": ["origin", "destination", "is_delayed", "transit_time_hours"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_corridor_live_weather",
            "description": "Fetches ambient weather and altitude pressure for key corridors (e.g., Beaufort West, Johannesburg, Durban).",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {"type": "string"}
                },
                "required": ["location_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_modal_shift_economics",
            "description": "Calculates carbon emission reductions and ZAR carbon tax savings for shifting road freight to rail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "corridor_key": {"type": "string"},
                    "shift_percentage": {"type": "number"},
                    "total_tkm": {"type": "number"}
                },
                "required": ["corridor_key", "shift_percentage"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are the AI Control Tower Copilot for a South African Cold Chain Logistics Network.
You assist operations managers with real-time fleet telemetry, predictive risk assessments, weather monitoring, and carbon calculations.

For general greetings or conversational prompts (e.g., 'hi', 'who are you', 'help'), reply directly and politely without calling any tools.
Use tools only when specific operational metrics, telemetry, risk scenarios, weather, or carbon shifts are requested.
"""


# --- 3. FASTAPI ROUTE ---

class AgentQueryRequest(BaseModel):
    user_query: str

class AgentQueryResponse(BaseModel):
    response: str


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(payload: AgentQueryRequest):
    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail="Groq Client not initialized. Set GROQ_API_KEY in Render environment."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": payload.user_query}
    ]

    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=GROQ_TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=800
        )
        response_msg = completion.choices[0].message

        # Handle tool calling
        if response_msg.tool_calls:
            messages.append(response_msg)

            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments or "{}")
                tool_func = AVAILABLE_TOOLS.get(func_name)

                if tool_func:
                    logger.info("Executing tool: %s with args: %s", func_name, func_args)
                    try:
                        tool_output = tool_func(**func_args)
                    except Exception as err:
                        tool_output = json.dumps({"error": f"Tool execution error: {str(err)}"})
                else:
                    tool_output = json.dumps({"error": f"Function {func_name} not found"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": tool_output
                })

            final_res = await groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=800
            )
            return AgentQueryResponse(response=final_res.choices[0].message.content or "")

        return AgentQueryResponse(response=response_msg.content or "Ready for your operations query.")

    except Exception as e:
        logger.exception("Agent execution failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
