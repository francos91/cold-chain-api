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

# Official fast production model on Groq
ACTIVE_MODEL = "openai/gpt-oss-20b"

if not GROQ_API_KEY:
  logger.warning("GROQ_API_KEY is not set in environment.")

try:
  supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
  logger.warning("Supabase client initialization deferred: %s", e)
  supabase = None

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Load containerized ML assets
try:
  rf_model = joblib.load("advanced_model.pkl")
  route_encoder = joblib.load("le_route.pkl")
  logger.info("Agent successfully loaded ML models.")
except Exception as e:
  logger.warning("Agent failed to load ML models: %s", e)
  rf_model, route_encoder = None, None

# Weather caching & regional telemetry safeguards
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


def get_active_fleet_telemetry(
    status_filter: str = "active",
    only_breaches: bool = False,
    **kwargs,
) -> str:
  """Fetches live telemetry for shipments from Supabase filtered by status and breaches."""
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

    res = query.order("temperature_celsius", desc=True).limit(15).execute()  # Reduced limit to conserve tokens
    return json.dumps(res.data or [])
  except Exception as e:
    return json.dumps({"error": f"Database query failed: {str(e)}"})


def get_fleet_aggregate_summary(**kwargs) -> str:
  """Calculates global fleet metrics across the entire master database."""
  if not supabase:
    return json.dumps({"error": "Supabase client is not connected."})
  try:
    res = (
        supabase.table("cold_chain_tracking")
        .select("shipment_id,origin,destination,status,temperature_celsius")
        .execute()
    )
    rows = res.data or []
    if not rows:
      return json.dumps({"error": "No records found in database."})

    df = pd.DataFrame(rows)
    total_records = len(df)
    status_counts = df["status"].value_counts().to_dict()

    top_destinations_all = df["destination"].value_counts().head(3).to_dict()
    top_origins_all = df["origin"].value_counts().head(3).to_dict()

    df_active = df[df["status"] != "delivered"]
    active_breaches = int((df_active["temperature_celsius"] > 8.0).sum())

    return json.dumps({
        "total_records_in_database": total_records,
        "status_distribution": status_counts,
        "active_fleet_count": len(df_active),
        "active_temperature_excursions": active_breaches,
        "top_overall_destinations": top_destinations_all,
        "top_overall_origins": top_origins_all,
    })
  except Exception as e:
    return json.dumps({"error": f"Fleet aggregation failed: {str(e)}"})


def predict_temperature_breach_risk(
    origin: str = "Johannesburg",
    destination: str = "Cape Town",
    is_delayed: bool = False,
    transit_time_hours: float = 24.0,
    distance_km: Any = None,
    **kwargs,
) -> str:
  """Runs the containerized Random Forest model to predict continuous probability of a temperature breach (>8°C)."""
  if rf_model is None or route_encoder is None:
    return json.dumps({"error": "ML models not available in backend container."})

  KNOWN_CORRIDOR_DISTANCES = {
      "polokwane_to_knysna": 1605.7,
      "knysna_to_polokwane": 1605.7,
      "johannesburg_to_durban": 570.0,
      "durban_to_johannesburg": 570.0,
      "stellenbosch_to_polokwane": 1894.7,
      "polokwane_to_stellenbosch": 1894.7,
      "george_to_polokwane": 1633.0,
      "polokwane_to_george": 1633.0,
      "johannesburg_to_cape_town": 1398.0,
      "cape_town_to_johannesburg": 1398.0,
      "cape_town_to_durban": 1635.0,
      "durban_to_cape_town": 1635.0,
  }

  clean_origin = str(origin).strip().lower().replace(" ", "_")
  clean_destination = str(destination).strip().lower().replace(" ", "_")
  route_key_normalized = f"{clean_origin}_to_{clean_destination}"
  route_key_title = f"{origin.strip().title().replace(' ', '_')}_to_{destination.strip().title().replace(' ', '_')}"

  try:
    effective_distance = float(distance_km) if distance_km is not None else 600.0
  except (ValueError, TypeError):
    effective_distance = 600.0

  if effective_distance == 600.0:
    effective_distance = KNOWN_CORRIDOR_DISTANCES.get(route_key_normalized, 600.0)

  route_enc = None
  for candidate in [route_key_normalized, route_key_title, f"{origin}_to_{destination}"]:
    try:
      route_enc = int(route_encoder.transform([candidate])[0])
      break
    except Exception:
      continue

  if route_enc is None:
    route_enc = len(route_encoder.classes_) // 2 if hasattr(route_encoder, "classes_") else 5

  base_risk = 0.50 if is_delayed else 0.10
  duration_penalty = min(0.35, (float(transit_time_hours) / 40.0) * 0.35)
  route_risk_proxy = round(min(0.95, base_risk + duration_penalty), 2)
  delayed_flag = 1 if is_delayed else 0

  features = pd.DataFrame([{
      "route_enc": route_enc,
      "route_risk_score": route_risk_proxy,
      "is_delayed": delayed_flag,
      "distance_km": effective_distance,
      "transit_time_hours": float(transit_time_hours),
  }])

  try:
    prob_raw = float(rf_model.predict_proba(features)[0, 1])
  except Exception:
    prob_raw = 0.0

  if is_delayed and float(transit_time_hours) >= 30.0:
    prob_raw = max(prob_raw, 0.74)
  elif is_delayed and float(transit_time_hours) >= 18.0:
    prob_raw = max(prob_raw, 0.46)
  elif is_delayed:
    prob_raw = max(prob_raw, 0.28)

  prob_pct = round(prob_raw * 100, 2)
  classification = "CRITICAL RISK" if prob_pct >= 40.0 else ("MODERATE RISK" if prob_pct >= 15.0 else "LOW RISK")

  return json.dumps({
      "route": f"{origin} to {destination}",
      "breach_probability_pct": prob_pct,
      "risk_classification": classification,
      "parameters": {
          "is_delayed": is_delayed,
          "transit_time_hours": transit_time_hours,
          "distance_km": effective_distance,
      },
  })


def get_corridor_live_weather(location_name: str = "Beaufort West", **kwargs) -> str:
  """Fetches ambient weather from Open-Meteo with caching and regional fallback."""
  COORDINATES = {
      "johannesburg": (-26.2041, 28.0473),
      "durban": (-29.8587, 31.0218),
      "cape town": (-33.9249, 18.4241),
      "beaufort west": (-32.3567, 22.5830),
      "bloemfontein": (-29.0852, 26.1596),
      "harrismith": (-28.2728, 29.1297),
      "polokwane": (-23.9045, 29.4688),
      "port elizabeth": (-33.9608, 25.6022),
      "george": (-33.9630, 22.4617),
      "mossel bay": (-34.1831, 22.1460),
      "knysna": (-34.0363, 23.0471),
  }

  loc_clean = str(location_name).strip().lower()
  now = time.time()

  if loc_clean in _WEATHER_CACHE:
    cached_entry, timestamp = _WEATHER_CACHE[loc_clean]
    if now - timestamp < _CACHE_TTL_SECONDS:
      return json.dumps(cached_entry)

  lat, lon = COORDINATES.get(loc_clean, (-32.3567, 22.5830))
  url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure"
  headers = {"User-Agent": "ColdChainLogisticsIntelligence/1.0"}

  try:
    r = requests.get(url, headers=headers, timeout=4)
    r.raise_for_status()
    current = r.json().get("current", {})

    weather_data = {
        "waypoint": location_name,
        "ambient_temp_celsius": current.get("temperature_2m"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "surface_pressure_hpa": current.get("surface_pressure", 1013.0),
        "source": "live_open_meteo",
    }
    _WEATHER_CACHE[loc_clean] = (weather_data, now)
    return json.dumps(weather_data)
  except Exception:
    fallback = REGIONAL_FALLBACK_WEATHER.get(loc_clean, {"temp": 25.0, "humidity": 50.0, "pressure": 950.0})
    return json.dumps({
        "waypoint": location_name,
        "ambient_temp_celsius": fallback["temp"],
        "humidity_pct": fallback["humidity"],
        "surface_pressure_hpa": fallback["pressure"],
        "source": "regional_fallback",
    })


def simulate_modal_shift_economics(
    corridor_key: str = "Natcor",
    shift_percentage: float = 30.0,
    total_tkm: float = 600000.0,
    **kwargs,
) -> str:
  """Calculates carbon reductions and South African Carbon Tax savings (R28.00/tonne)."""
  road_emission_factor = 73.2
  rail_emission_factor = 12.2
  effective_tax_rate_zar = 28.0

  baseline_emissions = (float(total_tkm) * road_emission_factor) / 1_000_000
  rail_tkm = float(total_tkm) * (float(shift_percentage) / 100.0)
  remaining_road_tkm = float(total_tkm) * (1 - (float(shift_percentage) / 100.0))

  new_emissions = ((remaining_road_tkm * road_emission_factor) + (rail_tkm * rail_emission_factor)) / 1_000_000
  carbon_saved_tco2 = baseline_emissions - new_emissions
  tax_savings_zar = carbon_saved_tco2 * effective_tax_rate_zar

  return json.dumps({
      "corridor": corridor_key,
      "modal_shift_pct": shift_percentage,
      "carbon_saved_tco2": round(carbon_saved_tco2, 2),
      "carbon_tax_savings_zar": round(tax_savings_zar, 2),
  })


AVAILABLE_TOOLS = {
    "get_active_fleet_telemetry": get_active_fleet_telemetry,
    "get_fleet_aggregate_summary": get_fleet_aggregate_summary,
    "predict_temperature_breach_risk": predict_temperature_breach_risk,
    "get_corridor_live_weather": get_corridor_live_weather,
    "simulate_modal_shift_economics": simulate_modal_shift_economics,
}

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_corridor_live_weather",
            "description": "Fetches ambient weather and pressure for freight corridors.",
            "parameters": {
                "type": "object",
                "properties": {"location_name": {"type": "string"}},
                "required": ["location_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fleet_aggregate_summary",
            "description": "Computes aggregate totals across the 4,500 master shipment database.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_fleet_telemetry",
            "description": "Fetches active shipment records and temperature excursions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status_filter": {"type": "string", "enum": ["active", "in-transit", "delayed", "delivered", "all"]},
                    "only_breaches": {"type": "boolean"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_temperature_breach_risk",
            "description": "Runs Random Forest model to predict thermal breach probability (>8°C).",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "is_delayed": {"type": "boolean"},
                    "transit_time_hours": {"type": "number"},
                    "distance_km": {"type": ["number", "null"]},
                },
                "required": ["origin", "destination", "is_delayed", "transit_time_hours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_modal_shift_economics",
            "description": "Calculates carbon and carbon tax savings for road-to-rail shifts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "corridor_key": {"type": "string"},
                    "shift_percentage": {"type": "number"},
                    "total_tkm": {"type": "number"},
                },
                "required": ["corridor_key", "shift_percentage"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are the AI Control Tower Copilot for South Africa's Cold Chain Logistics Network.
Be concise and direct to respect token rate limits. Format currency in ZAR (e.g., R28,000.00).
- For financial and carbon exposure questions, use standard baseline formulas or estimate based on fleet tkm.
- Never output token channel tags or raw code artifacts.
"""


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

  role_instructions = ""
  if payload.user_role == "Executive":
    role_instructions = "\nUSER ROLE: Executive. Keep answers brief, focus on total ZAR impact and OTIF."
  elif payload.user_role == "Dispatcher":
    role_instructions = "\nUSER ROLE: Dispatcher. Focus on specific shipment IDs and routes."

  messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT + role_instructions}]

  if payload.chat_history:
    for msg in payload.chat_history[-4:]:  # Reduced history length to save tokens
      if msg.role in ("user", "assistant") and msg.content and msg.content.strip():
        messages.append({"role": msg.role, "content": msg.content.strip()})

  messages.append({"role": "user", "content": payload.user_query})

  max_tool_iterations = 3
  iteration = 0

  try:
    while iteration < max_tool_iterations:
      iteration += 1

      # RATE LIMIT SAFEGUARD: Automatic Exponential Backoff Retry Loop
      retries = 3
      delay = 1.0
      completion = None

      for attempt in range(retries):
        try:
          completion = await groq_client.chat.completions.create(
              model=ACTIVE_MODEL,
              messages=messages,
              tools=GROQ_TOOLS,
              tool_choice="auto",
              temperature=0.1,
              max_tokens=600,  # Lower max tokens to stay safely below 8k TPM limit
          )
          break
        except RateLimitError as rle:
          if attempt == retries - 1:
            raise rle
          logger.warning("Groq TPM rate limit hit. Retrying in %.1fs...", delay)
          time.sleep(delay)
          delay *= 2.0

      if not completion:
        raise HTTPException(status_code=429, detail="Groq rate limit exceeded after retries.")

      response_msg = completion.choices[0].message

      if not response_msg.tool_calls:
        content = response_msg.content
        if content and content.strip():
          return AgentQueryResponse(response=content)
        return AgentQueryResponse(response="I am monitoring your active cold-chain fleet.")

      assistant_payload: Dict[str, Any] = {
          "role": "assistant",
          "tool_calls": [
              {
                  "id": tc.id,
                  "type": "function",
                  "function": {
                      "name": tc.function.name,
                      "arguments": tc.function.arguments,
                  },
              }
              for tc in response_msg.tool_calls
          ],
      }
      if response_msg.content:
        assistant_payload["content"] = response_msg.content

      messages.append(assistant_payload)

      for tool_call in response_msg.tool_calls:
        raw_func_name = tool_call.function.name
        
        # FIX TOOL NAME CORRUPTION: Strip any channel tags like <|channel|>commentary
        func_name = raw_func_name.split("<|")[0].strip()

        try:
          func_args = json.loads(tool_call.function.arguments or "{}")
        except Exception:
          func_args = {}

        tool_func = AVAILABLE_TOOLS.get(func_name)

        if tool_func:
          try:
            tool_output = tool_func(**func_args)
          except Exception as err:
            tool_output = json.dumps({"error": f"Tool error: {str(err)}"})
        else:
          tool_output = json.dumps({"error": f"Function '{func_name}' not recognized."})

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": func_name,
            "content": str(tool_output),
        })

    # Final synthesis call with exponential backoff retry (Bumped max_tokens to 1000 to prevent response cutoff)
    final_completion = None
    delay = 1.0
    for attempt in range(3):
      try:
        final_completion = await groq_client.chat.completions.create(
            model=ACTIVE_MODEL, messages=messages, temperature=0.1, max_tokens=1000
        )
        break
      except RateLimitError:
        if attempt == 2:
          raise
        time.sleep(delay)
        delay *= 2.0

    return AgentQueryResponse(
        response=final_completion.choices[0].message.content or "Completed multi-step audit."
    )

  except RateLimitError as rle:
    logger.error("Groq rate limit exhausted: %s", str(rle))
    raise HTTPException(status_code=429, detail="Groq token rate limit (8,000 TPM) reached. Please wait 10 seconds and try again.")
  except Exception as e:
    logger.exception("Agent execution failed: %s", str(e))
    raise HTTPException(status_code=500, detail=str(e))
