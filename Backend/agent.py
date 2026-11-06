import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from groq import AsyncGroq, RateLimitError
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import requests
from supabase import Client, create_client

logger = logging.getLogger("coldchain-agent")
router = APIRouter(prefix="/api/v1/agent", tags=["AI Operations Copilot"])

# 1. Environment variables with CI fallbacks
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://placeholder.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "placeholder-key")

ACTIVE_MODEL = "openai/gpt-oss-20b"

if not GROQ_API_KEY:
  logger.warning("GROQ_API_KEY is not set in environment.")

try:
  supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
  logger.warning("Supabase client initialization deferred: %s", e)
  supabase = None

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

try:
  rf_model = joblib.load("advanced_model.pkl")
  route_encoder = joblib.load("le_route.pkl")
  logger.info("Agent successfully loaded ML models.")
except Exception as e:
  logger.warning("Agent failed to load ML models: %s", e)
  rf_model, route_encoder = None, None

_WEATHER_CACHE: Dict[str, Any] = {}
_CACHE_TTL_SECONDS = 600

REGIONAL_FALLBACK_WEATHER = {
    "beaufort west": {"temp": 28.5, "humidity": 22.0, "pressure": 918.0},
    "johannesburg": {"temp": 24.0, "humidity": 45.0, "pressure": 825.0},
    "durban": {"temp": 26.5, "humidity": 78.0, "pressure": 1012.0},
    "cape town": {"temp": 22.0, "humidity": 65.0, "pressure": 1014.0},
    "bloemfontein": {"temp": 27.0, "humidity": 30.0, "pressure": 860.0},
    "harrismith": {"temp": 21.0, "humidity": 50.0, "pressure": 830.0},
    "polokwane": {"temp": 26.0, "humidity": 40.0, "pressure": 870.0},
    "port elizabeth": {"temp": 23.0, "humidity": 70.0, "pressure": 1012.0},
    "george": {"temp": 21.5, "humidity": 75.0, "pressure": 995.0},
    "mossel bay": {"temp": 21.0, "humidity": 72.0, "pressure": 1010.0},
    "knysna": {"temp": 22.0, "humidity": 76.0, "pressure": 1008.0},
}


# --- 2. EXECUTABLE TOOLS ---


def get_active_fleet_telemetry(status_filter: str = "active", only_breaches: bool = False, **kwargs) -> str:
  if not supabase:
    return json.dumps({"error": "Supabase client is not connected."})
  try:
    query = supabase.table("cold_chain_tracking").select(
        "shipment_id,origin,destination,temperature_celsius,status,distance_km,transit_time_hours,weight_tonnes"
    )
    clean_status = str(status_filter).strip().lower()
    if clean_status == "delivered":
      query = query.eq("status", "delivered")
    elif clean_status == "delayed":
      query = query.eq("status", "delayed")
    elif clean_status in ("in-transit", "in_transit"):
      query = query.eq("status", "in-transit")
    elif clean_status == "active":
      query = query.not_.eq("status", "delivered")

    if only_breaches:
      query = query.gt("temperature_celsius", 8.0)

    res = query.order("temperature_celsius", desc=True).limit(10).execute()
    return json.dumps(res.data or [])
  except Exception as e:
    return json.dumps({"error": f"Database query failed: {str(e)}"})


def get_fleet_aggregate_summary(**kwargs) -> str:
  if not supabase:
    return json.dumps({"error": "Supabase client is not connected."})
  try:
    res = supabase.table("cold_chain_tracking").select("shipment_id,origin,destination,status,temperature_celsius").execute()
    rows = res.data or []
    if not rows:
      return json.dumps({"error": "No records found."})

    df = pd.DataFrame(rows)
    total_records = len(df)
    status_counts = df["status"].value_counts().to_dict()
    top_destinations_all = df["destination"].value_counts().head(3).to_dict()
    top_origins_all = df["origin"].value_counts().head(3).to_dict()
    df_active = df[df["status"] != "delivered"]
    active_breaches = int((df_active["temperature_celsius"] > 8.0).sum())

    return json.dumps({
        "total_records": total_records,
        "status_distribution": status_counts,
        "active_count": len(df_active),
        "active_excursions": active_breaches,
        "top_destinations": top_destinations_all,
        "top_origins": top_origins_all,
    })
  except Exception as e:
    return json.dumps({"error": f"Aggregation failed: {str(e)}"})


def predict_temperature_breach_risk(
    origin: str = "Johannesburg",
    destination: str = "Cape Town",
    is_delayed: bool = False,
    transit_time_hours: float = 24.0,
    distance_km: Any = None,
    **kwargs,
) -> str:
  if rf_model is None or route_encoder is None:
    return json.dumps({"error": "ML models not available."})

  KNOWN_CORRIDOR_DISTANCES = {
      "johannesburg_to_durban": 570.0,
      "durban_to_johannesburg": 570.0,
      "johannesburg_to_cape_town": 1398.0,
      "cape_town_to_johannesburg": 1398.0,
  }

  clean_origin = str(origin).strip().lower().replace(" ", "_")
  clean_destination = str(destination).strip().lower().replace(" ", "_")
  route_key = f"{clean_origin}_to_{clean_destination}"

  try:
    effective_distance = float(distance_km) if distance_km is not None else KNOWN_CORRIDOR_DISTANCES.get(route_key, 600.0)
  except (ValueError, TypeError):
    effective_distance = 600.0

  route_enc = 5
  try:
    route_enc = int(route_encoder.transform([route_key])[0])
  except Exception:
    pass

  base_risk = 0.50 if is_delayed else 0.10
  duration_penalty = min(0.35, (float(transit_time_hours) / 40.0) * 0.35)
  route_risk_proxy = round(min(0.95, base_risk + duration_penalty), 2)

  features = pd.DataFrame([{
      "route_enc": route_enc,
      "route_risk_score": route_risk_proxy,
      "is_delayed": 1 if is_delayed else 0,
      "distance_km": effective_distance,
      "transit_time_hours": float(transit_time_hours),
  }])

  try:
    prob_raw = float(rf_model.predict_proba(features)[0, 1])
  except Exception:
    prob_raw = 0.0

  if is_delayed and float(transit_time_hours) >= 30.0:
    prob_raw = max(prob_raw, 0.74)

  prob_pct = round(prob_raw * 100, 2)
  classification = "CRITICAL RISK" if prob_pct >= 40.0 else ("MODERATE RISK" if prob_pct >= 15.0 else "LOW RISK")

  return json.dumps({
      "route": f"{origin} to {destination}",
      "breach_probability_pct": prob_pct,
      "risk_classification": classification,
  })


def get_corridor_live_weather(location_name: str = "Beaufort West", **kwargs) -> str:
  loc_clean = str(location_name).strip().lower()
  now = time.time()
  if loc_clean in _WEATHER_CACHE:
    cached_entry, timestamp = _WEATHER_CACHE[loc_clean]
    if now - timestamp < _CACHE_TTL_SECONDS:
      return json.dumps(cached_entry)

  coords = {"johannesburg": (-26.2, 28.04), "durban": (-29.85, 31.02), "cape town": (-33.92, 18.42), "beaufort west": (-32.35, 22.58)}
  lat, lon = coords.get(loc_clean, (-32.35, 22.58))
  url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure"

  try:
    r = requests.get(url, headers={"User-Agent": "ColdChainApp/1.0"}, timeout=3)
    r.raise_for_status()
    current = r.json().get("current", {})
    weather_data = {
        "waypoint": location_name,
        "ambient_temp_celsius": current.get("temperature_2m"),
        "humidity_pct": current.get("relative_humidity_2m"),
    }
    _WEATHER_CACHE[loc_clean] = (weather_data, now)
    return json.dumps(weather_data)
  except Exception:
    return json.dumps({"waypoint": location_name, "ambient_temp_celsius": 25.0, "source": "fallback"})


def simulate_modal_shift_economics(corridor_key: str = "Natcor", shift_percentage: float = 30.0, total_tkm: float = 600000.0, **kwargs) -> str:
  base = (float(total_tkm) * 73.2) / 1_000_000
  new = ((float(total_tkm) * (1 - shift_percentage / 100.0)) * 73.2 + (float(total_tkm) * (shift_percentage / 100.0) * 12.2)) / 1_000_000
  saved = base - new
  return json.dumps({"carbon_saved_tco2": round(saved, 2), "carbon_tax_savings_zar": round(saved * 28.0, 2)})


AVAILABLE_TOOLS = {
    "get_active_fleet_telemetry": get_active_fleet_telemetry,
    "get_fleet_aggregate_summary": get_fleet_aggregate_summary,
    "predict_temperature_breach_risk": predict_temperature_breach_risk,
    "get_corridor_live_weather": get_corridor_live_weather,
    "simulate_modal_shift_economics": simulate_modal_shift_economics,
}

GROQ_TOOLS = [
    {"type": "function", "function": {"name": "get_corridor_live_weather", "description": "Get weather for corridor waypoint.", "parameters": {"type": "object", "properties": {"location_name": {"type": "string"}}, "required": ["location_name"]}}},
    {"type": "function", "function": {"name": "get_fleet_aggregate_summary", "description": "Get overall fleet statistics.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_active_fleet_telemetry", "description": "Get active shipments.", "parameters": {"type": "object", "properties": {"status_filter": {"type": "string"}, "only_breaches": {"type": "boolean"}}, "required": []}}},
    {"type": "function", "function": {"name": "predict_temperature_breach_risk", "description": "Run ML breach risk model.", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "is_delayed": {"type": "boolean"}, "transit_time_hours": {"type": "number"}, "distance_km": {"type": ["number", "null"]}}, "required": ["origin", "destination", "is_delayed", "transit_time_hours"]}}},
    {"type": "function", "function": {"name": "simulate_modal_shift_economics", "description": "Calculate carbon tax savings.", "parameters": {"type": "object", "properties": {"corridor_key": {"type": "string"}, "shift_percentage": {"type": "number"}, "total_tkm": {"type": "number"}}, "required": ["corridor_key", "shift_percentage"]}}},
]

SYSTEM_PROMPT = "You are the AI Control Tower Copilot for South Africa Cold Chain. Be concise. Format currency in ZAR (e.g. R28,000)."


class ChatMessage(BaseModel):
  role: str
  content: str


class AgentQueryRequest(BaseModel):
  user_query: str
  chat_history: Optional[List[ChatMessage]] = Field(default_factory=list)
  user_role: Optional[str] = "Operations Manager"


class AgentQueryResponse(BaseModel):
  response: str


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(payload: AgentQueryRequest):
  if not groq_client:
    raise HTTPException(status_code=503, detail="Groq Client not initialized.")

  role_inst = " Executive: keep brief, focus on ZAR & OTIF." if payload.user_role == "Executive" else ""
  messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT + role_inst}]

  # Truncate history to last 2 messages to strictly preserve token budget
  if payload.chat_history:
    for msg in payload.chat_history[-2:]:
      if msg.role in ("user", "assistant") and msg.content:
        messages.append({"role": msg.role, "content": msg.content.strip()})

  messages.append({"role": "user", "content": payload.user_query})

  try:
    for iteration in range(2):  # Max 2 tool iterations to conserve tokens
      # RATE LIMIT SAFEGUARD: Longer backoff (8s) to clear rolling 60s window
      completion = None
      delay = 8.0
      for attempt in range(3):
        try:
          completion = await groq_client.chat.completions.create(
              model=ACTIVE_MODEL,
              messages=messages,
              tools=GROQ_TOOLS,
              tool_choice="auto",
              temperature=0.1,
              max_tokens=400,
          )
          break
        except RateLimitError as rle:
          if attempt == 2:
            raise rle
          logger.warning("Groq rate limit hit. Waiting %.1fs for rolling window...", delay)
          time.sleep(delay)
          delay *= 1.5

      if not completion:
        raise HTTPException(status_code=429, detail="Groq TPM limit reached. Please wait 15 seconds before trying again.")

      response_msg = completion.choices[0].message
      if not response_msg.tool_calls:
        return AgentQueryResponse(response=response_msg.content or "Fleet check complete.")

      messages.append({
          "role": "assistant",
          "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in response_msg.tool_calls]
      })

      for tc in response_msg.tool_calls:
        func_name = tc.function.name.split("<|")[0].strip()
        try:
          args = json.loads(tc.function.arguments or "{}")
        except Exception:
          args = {}

        tool_func = AVAILABLE_TOOLS.get(func_name)
        output = tool_func(**args) if tool_func else json.dumps({"error": "Tool not found"})

        messages.append({"role": "tool", "tool_call_id": tc.id, "name": func_name, "content": str(output)})

    # Final synthesis call
    final_completion = None
    delay = 8.0
    for attempt in range(3):
      try:
        final_completion = await groq_client.chat.completions.create(
            model=ACTIVE_MODEL, messages=messages, temperature=0.1, max_tokens=700
        )
        break
      except RateLimitError:
        if attempt == 2:
          raise
        time.sleep(delay)
        delay *= 1.5

    return AgentQueryResponse(response=final_completion.choices[0].message.content or "Audit complete.")

  except RateLimitError:
    raise HTTPException(status_code=429, detail="Groq token rate limit (8k TPM) reached. Please wait 15 seconds and try again.")
  except Exception as e:
    logger.exception("Agent execution failed: %s", str(e))
    raise HTTPException(status_code=500, detail=str(e))
