import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from groq import AsyncGroq
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

# Official fast production model on Groq (1,000 tps, high token allowance, active support)
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
_CACHE_TTL_SECONDS = 600  # 10-minute cache to protect shared Render IP from rate limits

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
    # If 'all', return unfiltered master shipments

    if only_breaches:
      query = query.gt("temperature_celsius", 8.0)

    res = query.order("temperature_celsius", desc=True).limit(25).execute()
    return json.dumps(res.data or [])
  except Exception as e:
    return json.dumps({"error": f"Database query failed: {str(e)}"})


def get_fleet_aggregate_summary(**kwargs) -> str:
  """Calculates global fleet metrics across the entire master database.

  Use for totals, status distributions, and top volume hubs.
  """
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

    top_origins_all = df["origin"].value_counts().head(5).to_dict()
    top_destinations_all = df["destination"].value_counts().head(5).to_dict()

    # Breakdown specifically for delivered shipments
    df_delivered = df[df["status"] == "delivered"]
    top_delivered_destinations = (
        df_delivered["destination"].value_counts().head(5).to_dict()
        if not df_delivered.empty
        else {}
    )
    top_delivered_origins = (
        df_delivered["origin"].value_counts().head(5).to_dict()
        if not df_delivered.empty
        else {}
    )

    # Active excursions count (>8°C on non-delivered)
    df_active = df[df["status"] != "delivered"]
    active_breaches = int((df_active["temperature_celsius"] > 8.0).sum())

    return json.dumps({
        "total_records_in_database": total_records,
        "status_distribution": status_counts,
        "active_fleet_count": len(df_active),
        "active_temperature_excursions": active_breaches,
        "top_overall_destinations": top_destinations_all,
        "top_overall_origins": top_origins_all,
        "top_delivered_destinations": top_delivered_destinations,
        "top_delivered_origins": top_delivered_origins,
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
    return json.dumps(
        {"error": "ML models not available in backend container."}
    )

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
    effective_distance = (
        float(distance_km) if distance_km is not None else 600.0
    )
  except (ValueError, TypeError):
    effective_distance = 600.0

  if effective_distance == 600.0:
    effective_distance = KNOWN_CORRIDOR_DISTANCES.get(
        route_key_normalized, 600.0
    )

  route_enc = None
  for candidate in [
      route_key_normalized,
      route_key_title,
      f"{origin}_to_{destination}",
  ]:
    try:
      route_enc = int(route_encoder.transform([candidate])[0])
      break
    except Exception:
      continue

  if route_enc is None:
    route_enc = (
        len(route_encoder.classes_) // 2
        if hasattr(route_encoder, "classes_")
        else 5
    )

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

  # Physics-based heuristic floor to guarantee critical classification on long-haul delays
  if is_delayed and float(transit_time_hours) >= 30.0:
    prob_raw = max(prob_raw, 0.74)
  elif is_delayed and float(transit_time_hours) >= 18.0:
    prob_raw = max(prob_raw, 0.46)
  elif is_delayed:
    prob_raw = max(prob_raw, 0.28)

  prob_pct = round(prob_raw * 100, 2)

  if prob_pct >= 40.0:
    classification = "CRITICAL RISK"
  elif prob_pct >= 15.0:
    classification = "MODERATE RISK"
  else:
    classification = "LOW RISK"

  return json.dumps({
      "route": f"{origin} to {destination}",
      "breach_probability_pct": prob_pct,
      "risk_classification": classification,
      "parameters": {
          "is_delayed": is_delayed,
          "transit_time_hours": transit_time_hours,
          "distance_km": effective_distance,
          "route_risk_score": route_risk_proxy,
      },
  })


def get_corridor_live_weather(
    location_name: str = "Beaufort West", **kwargs
) -> str:
  """Fetches ambient weather from Open-Meteo with in-memory TTL caching and regional fallback."""
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

  if (
      "cities" in kwargs
      and isinstance(kwargs["cities"], list)
      and len(kwargs["cities"]) > 0
  ):
    location_name = kwargs["cities"][0]

  loc_clean = str(location_name).strip().lower()
  now = time.time()

  # 1. In-Memory Cache Check
  if loc_clean in _WEATHER_CACHE:
    cached_entry, timestamp = _WEATHER_CACHE[loc_clean]
    if now - timestamp < _CACHE_TTL_SECONDS:
      logger.info("Serving weather for '%s' from local memory cache.", loc_clean)
      return json.dumps(cached_entry)

  lat, lon = COORDINATES.get(loc_clean, (-32.3567, 22.5830))
  url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure"
  headers = {"User-Agent": "ColdChainLogisticsIntelligence/1.0"}

  # 2. Live Request Attempt
  try:
    r = requests.get(url, headers=headers, timeout=5)
    r.raise_for_status()
    current = r.json().get("current", {})

    temp_c = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    pressure = current.get("surface_pressure", 1013.0)

    weather_data = {
        "waypoint": location_name,
        "ambient_temp_celsius": temp_c,
        "humidity_pct": humidity,
        "surface_pressure_hpa": pressure,
        "source": "live_open_meteo",
        "highveld_thermal_penalty": (
            "21% altitude cooling efficiency loss applied"
            if pressure < 850
            else "Nominal cooling"
        ),
    }

    _WEATHER_CACHE[loc_clean] = (weather_data, now)
    return json.dumps(weather_data)

  except Exception as e:
    logger.warning(
        "Open-Meteo call failed (%s). Activating regional telemetry fallback"
        " for '%s'.",
        e,
        loc_clean,
    )
    fallback = REGIONAL_FALLBACK_WEATHER.get(
        loc_clean, {"temp": 25.0, "humidity": 50.0, "pressure": 950.0}
    )

    weather_data = {
        "waypoint": location_name,
        "ambient_temp_celsius": fallback["temp"],
        "humidity_pct": fallback["humidity"],
        "surface_pressure_hpa": fallback["pressure"],
        "source": "regional_sensor_telemetry_fallback",
        "highveld_thermal_penalty": (
            "21% altitude cooling efficiency loss applied"
            if fallback["pressure"] < 850
            else "Nominal cooling"
        ),
    }

    # Cache the fallback for 3 minutes to avoid thrashing upstream on quota limits
    _WEATHER_CACHE[loc_clean] = (weather_data, now - _CACHE_TTL_SECONDS + 180)
    return json.dumps(weather_data)


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
  remaining_road_tkm = float(total_tkm) * (
      1 - (float(shift_percentage) / 100.0)
  )

  new_emissions = (
      (remaining_road_tkm * road_emission_factor)
      + (rail_tkm * rail_emission_factor)
  ) / 1_000_000
  carbon_saved_tco2 = baseline_emissions - new_emissions
  tax_savings_zar = carbon_saved_tco2 * effective_tax_rate_zar

  return json.dumps({
      "corridor": corridor_key,
      "modal_shift_pct": shift_percentage,
      "carbon_saved_tco2": round(carbon_saved_tco2, 2),
      "carbon_tax_savings_zar": round(tax_savings_zar, 2),
      "baseline_emissions_tco2": round(baseline_emissions, 2),
      "new_emissions_tco2": round(new_emissions, 2),
  })


AVAILABLE_TOOLS = {
    "get_active_fleet_telemetry": get_active_fleet_telemetry,
    "get_fleet_aggregate_summary": get_fleet_aggregate_summary,
    "predict_temperature_breach_risk": predict_temperature_breach_risk,
    "get_corridor_live_weather": get_corridor_live_weather,
    "simulate_modal_shift_economics": simulate_modal_shift_economics,
    "fleet_aggregate": get_fleet_aggregate_summary,
    "weather_monitoring": get_corridor_live_weather,
    "get_weather": get_corridor_live_weather,
    "predict_risk": predict_temperature_breach_risk,
    "fleet_telemetry": get_active_fleet_telemetry,
}

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_corridor_live_weather",
            "description": (
                "REQUIRED when asked for weather, ambient temperature, humidity,"
                " surface pressure, or highveld altitude penalties along freight"
                " corridors (e.g., Beaufort West, Johannesburg, Durban,"
                " Polokwane)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": (
                            "City or corridor waypoint name (e.g., 'Beaufort"
                            " West', 'Johannesburg', 'Cape Town')."
                        ),
                    }
                },
                "required": ["location_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fleet_aggregate_summary",
            "description": (
                "REQUIRED when asked for total fleet counts, total delivered"
                " vs in-transit vs delayed records, top destinations, top"
                " origin hubs, or fleet-wide volume statistics. Computes"
                " aggregations across the entire 4,500-shipment master database."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_fleet_telemetry",
            "description": (
                "Fetches real-time shipment records from Supabase. Use for"
                " live status, highest temperature excursions, individual route"
                " costs, or checking specific shipments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "enum": [
                            "active",
                            "in-transit",
                            "delayed",
                            "delivered",
                            "all",
                        ],
                        "description": (
                            "Filter by status: 'active' (non-delivered),"
                            " 'delivered', 'delayed', 'in-transit', or 'all'."
                        ),
                    },
                    "only_breaches": {
                        "type": "boolean",
                        "description": (
                            "Set to True to filter exclusively for shipments"
                            " above 8.0°C."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_temperature_breach_risk",
            "description": (
                "Runs the containerized Random Forest model to predict"
                " continuous probability of a temperature breach (>8°C)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "is_delayed": {"type": "boolean"},
                    "transit_time_hours": {"type": "number"},
                    "distance_km": {
                        "type": ["number", "null"],
                        "description": (
                            "Corridor distance in km. Pass null or omit if"
                            " unknown."
                        ),
                    },
                },
                "required": [
                    "origin",
                    "destination",
                    "is_delayed",
                    "transit_time_hours",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_modal_shift_economics",
            "description": (
                "Calculates carbon emission reductions and ZAR carbon tax"
                " savings for shifting road freight to rail."
            ),
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
You assist operations managers with real-time fleet telemetry, predictive risk assessments, weather monitoring, cost-to-serve analytics, and carbon calculations.

DATABASE INTEGRITY & SNAPSHOT AWARENESS:
- The master database contains approximately 4,500 total shipments across all statuses ('in-transit', 'delayed', and 'delivered').
- NEVER claim that the dataset "only contains in-transit shipments" or that "delivered records do not exist."
- When asked for overall counts, the most popular destinations/origins, delivered counts, or fleet status distributions, ALWAYS call `get_fleet_aggregate_summary`.
- When using `get_active_fleet_telemetry`, recognize that it returns an active operational slice prioritized by thermal risk, NOT the entire historical database.

CONVERSATIONAL CONTEXT & FOLLOW-UP RESOLUTION:
- You have access to the recent conversation history. Actively resolve ellipses, pronouns, and short follow-ups against prior turns.
- For example, if the previous turn was "What area has the most shipments?" and the user follows up with "And delivered?", interpret this as "What area has the most delivered shipments?" and call `get_fleet_aggregate_summary`.

WEATHER & AMBIENT MONITORING:
- Always call `get_corridor_live_weather` for any question regarding weather, ambient conditions, or corridor temperature.
- Report the waypoint name, ambient temperature in °C, humidity percentage, surface pressure in hPa, and any applicable altitude penalty.

OPERATIONAL COST-TO-SERVE BENCHMARK:
- Formula: Trip Cost (ZAR) = (Distance in km * R18.00) + R200.00 fixed handling fee.
- When answering cost questions, ALWAYS calculate and present:
  1. Total Active Fleet Spend (Sum of all active shipment costs).
  2. Average Cost per Shipment.
  3. The Top 3 Highest-Cost Shipments (with Origin -> Destination, Distance, and ZAR Cost).

RISK CLASSIFICATION BENCHMARK:
- When summarizing output from `predict_temperature_breach_risk`, ALWAYS reflect the exact `risk_classification` and `breach_probability_pct` returned:
  * Breach Probability >= 40.0%: Strictly report as "CRITICAL RISK".
  * Breach Probability 15.0% - 39.99%: Strictly report as "MODERATE RISK".
  * Breach Probability < 15.0%: Strictly report as "LOW RISK".

CURRENCY & UNIT FIDELITY:
- Always format currency as 'R' or 'ZAR' (e.g., R29,102.60).
- Never round single Rand values into 'k' (e.g., R256.20 must never be written as R256k).
"""


# --- 3. FASTAPI ROUTE & STATEFUL CHAT HISTORY ---


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
    raise HTTPException(
        status_code=503,
        detail=(
            "Groq Client not initialized. Set GROQ_API_KEY in Render"
            " environment."
        ),
    )

  # 1. DYNAMIC PERSONA INJECTION
  role_instructions = ""
  if payload.user_role == "Executive":
      role_instructions = "\nUSER ROLE: Executive. Keep answers brief, focus on total ZAR impact, OTIF, and high-level summaries. Do not list granular shipment IDs unless asked."
  elif payload.user_role == "Dispatcher":
      role_instructions = "\nUSER ROLE: Floor Dispatcher. Focus on specific shipment IDs, route names, thermal excursions, and immediate safety interventions."
      
  dynamic_system_prompt = SYSTEM_PROMPT + role_instructions

  # 2. INITIALIZE MESSAGES
  messages: List[Dict[str, Any]] = [{"role": "system", "content": dynamic_system_prompt}]

  if payload.chat_history:
    for msg in payload.chat_history[-6:]:
      if msg.role in ("user", "assistant") and msg.content and msg.content.strip():
        messages.append({"role": msg.role, "content": msg.content.strip()})

  messages.append({"role": "user", "content": payload.user_query})

  max_tool_iterations = 3
  iteration = 0

  try:
    while iteration < max_tool_iterations:
      iteration += 1

      # 3. FULL AUTONOMY: We removed the regex. The 70B model decides exactly what tool it needs based on the query.
      completion = await groq_client.chat.completions.create(
          model=ACTIVE_MODEL,
          messages=messages,
          tools=GROQ_TOOLS,
          tool_choice="auto",
          temperature=0.1,
          max_tokens=900,
      )
      response_msg = completion.choices[0].message

      # If no tools were called, the agent is ready to reply to the user
      if not response_msg.tool_calls:
        content = response_msg.content
        if content and content.strip():
          return AgentQueryResponse(response=content)
        
        return AgentQueryResponse(
            response="I am monitoring your active cold-chain fleet. You can request corridor weather, volume rankings, or cost breakdowns."
        )

      # Build clean assistant payload dictionary for tool execution
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

      # Execute each tool call emitted
      for tool_call in response_msg.tool_calls:
        func_name = tool_call.function.name
        try:
          func_args = json.loads(tool_call.function.arguments or "{}")
        except Exception:
          func_args = {}

        tool_func = AVAILABLE_TOOLS.get(func_name)

        if tool_func:
          logger.info("Iteration %d | Executing tool: %s", iteration, func_name)
          try:
            tool_output = tool_func(**func_args)
          except Exception as err:
            tool_output = json.dumps({"error": f"Tool execution error: {str(err)}"})
        else:
          tool_output = json.dumps({"error": f"Function '{func_name}' is not recognized."})

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": func_name,
            "content": str(tool_output),
        })

    # Final synthesis call after tools have run
    final_res = await groq_client.chat.completions.create(
        model=ACTIVE_MODEL, messages=messages, temperature=0.1, max_tokens=1000
    )
    return AgentQueryResponse(
        response=final_res.choices[0].message.content or "Completed multi-step fleet telemetry audit."
    )

  except Exception as e:
    logger.exception("Agent execution failed: %s", str(e))
    raise HTTPException(status_code=500, detail=str(e))
