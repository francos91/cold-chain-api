from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from groq import AsyncGroq, RateLimitError
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import requests
from supabase import Client, create_client

try:
  from carbon_calculator import calculate_cold_chain_carbon
except ImportError:
  from Backend.carbon_calculator import calculate_cold_chain_carbon

logger = logging.getLogger("coldchain-agent")
router = APIRouter(prefix="/api/v1/agent", tags=["AI Operations Copilot"])

# 1. Environment variables with CI fallbacks
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://placeholder.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "placeholder-key")

# Official fast production models on Groq with automatic 429 rate-limit failover
PRIMARY_MODEL = "openai/gpt-oss-20b"
FALLBACK_MODELS = ["openai/gpt-oss-120b"]
ACTIVE_MODEL = PRIMARY_MODEL

if not GROQ_API_KEY:
  logger.warning("GROQ_API_KEY is not set in environment.")
if not GEMINI_API_KEY:
  logger.warning(
      "GEMINI_API_KEY is not set in environment — RAG vector search disabled."
  )

try:
  supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
  logger.warning("Supabase client initialization deferred: %s", e)
  supabase = None

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


async def _create_completion_with_fallback(**kwargs):
  """Tries PRIMARY_MODEL first and fails over to FALLBACK_MODELS on Groq 429 rate limits."""
  models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS
  last_err = None

  for model_id in models_to_try:
    try:
      return await groq_client.chat.completions.create(
          model=model_id, **kwargs
      )
    except RateLimitError as e:
      logger.warning(
          "Groq 429 rate limit hit on '%s'. Failing over to next model...",
          model_id,
      )
      last_err = e
      continue

  raise last_err


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
_CACHE_TTL_SECONDS = (
    600  # 10-minute cache to protect shared Render IP from rate limits
)
_EMBED_CACHE: Dict[str, List[float]] = {}

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


def _normalize_vec(values: List[float]) -> List[float]:
  norm = math.sqrt(sum(x * x for x in values))
  return [x / norm for x in values] if norm > 0 else values


def _embed_query_gemini(query: str) -> List[float]:
  """Embeds a user search query with Gemini Embedding 001 (768-dim L2-normalized)."""
  cache_key = query.strip().lower()
  if cache_key in _EMBED_CACHE:
    return _EMBED_CACHE[cache_key]

  if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in Render environment.")

  url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
  payload = {
      "model": "models/gemini-embedding-001",
      "content": {"parts": [{"text": query}]},
      "taskType": "RETRIEVAL_QUERY",
      "outputDimensionality": 768,
  }
  r = requests.post(url, json=payload, timeout=5)
  r.raise_for_status()
  vec = _normalize_vec(r.json()["embedding"]["values"])

  if len(_EMBED_CACHE) >= 64:
    _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))
  _EMBED_CACHE[cache_key] = vec
  return vec


def search_cold_chain_knowledge(
    query: str,
    category: Optional[str] = None,
    top_k: int = 2,
    **kwargs,
) -> str:
  """Searches the Hybrid RRF (Semantic Vector + Full-Text Keyword) cold-chain knowledge base."""
  if not supabase:
    return json.dumps({"error": "Supabase client is not connected."})
  if not query or not str(query).strip():
    return json.dumps({"error": "Search query cannot be empty."})

  clean_q = str(query).strip()[:500]
  valid_categories = {
      "retail_and_industry_reports",
      "climate_policy_and_carbon",
      "thermal_and_sops",
      "government_and_industry",
      "meng_coursework_and_assignments",
      "general",
  }
  clean_cat = (
      category.strip()
      if category and str(category).strip() in valid_categories
      else None
  )
  # Strictly capped at 3 to stay safely within Groq Free Tier TPM limits
  safe_k = max(1, min(int(top_k or 2), 3))

  try:
    query_vec = _embed_query_gemini(clean_q)
    rpc_params = {
        "query_embedding": query_vec,
        "query_text": clean_q,
        "match_threshold": 0.30,
        "match_count": safe_k,
        "filter_category": clean_cat,
    }
    res = supabase.rpc("match_cold_chain_docs", rpc_params).execute()
    rows = res.data or []

    if not rows and clean_cat is not None:
      # Retry across all categories if strict category filter yielded zero matches
      rpc_params["filter_category"] = None
      res = supabase.rpc("match_cold_chain_docs", rpc_params).execute()
      rows = res.data or []

    if not rows:
      return json.dumps({
          "matches": [],
          "message": "No matching passages found in the knowledge base.",
      })

    formatted_matches = [
        {
            "source_file": row.get("source_file"),
            "category": row.get("category"),
            "page_number": row.get("page_number"),
            "similarity": round(float(row.get("similarity", 0.0)), 3),
            "excerpt": row.get("content", "")[
                :300
            ],  # Truncate excerpt for token efficiency
        }
        for row in rows
    ]
    return json.dumps({"matches": formatted_matches})
  except Exception as e:
    logger.warning("Knowledge base retrieval failed: %s", e)
    return json.dumps({"error": f"Knowledge base search failed: {str(e)}"})


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


def run_cold_chain_carbon_accounting(
    distance_km: float,
    cargo_weight_tonnes: float,
    temperature_regime: str = "chilled",
    refrigeration_hours: Optional[float] = None,
    route_direction: str = "to_coast",
    trailer_type: str = "tri_axle_reefer",
    empty_distance_km: Optional[float] = None,
    empty_running_pct: float = 0.0,
    average_speed_kmh: float = 65.0,
    dwell_and_precool_hours: float = 0.0,
    distribution_loss_pct: float = 0.0,
    **kwargs,
) -> str:
  """Executes the deterministic Stellenbosch & GLEC v3.1 carbon accounting engine."""
  try:
    result = calculate_cold_chain_carbon(
        distance_km=float(distance_km),
        cargo_weight_tonnes=float(cargo_weight_tonnes),
        temperature_regime=str(temperature_regime).strip().lower(),
        refrigeration_hours=(
            float(refrigeration_hours)
            if refrigeration_hours is not None
            else None
        ),
        route_direction=str(route_direction).strip().lower(),
        trailer_type=str(trailer_type).strip().lower(),
        empty_distance_km=(
            float(empty_distance_km) if empty_distance_km is not None else None
        ),
        empty_running_pct=float(empty_running_pct or 0.0),
        average_speed_kmh=float(average_speed_kmh or 65.0),
        dwell_and_precool_hours=float(dwell_and_precool_hours or 0.0),
        distribution_loss_pct=float(distribution_loss_pct or 0.0),
    )
    return json.dumps(result)
  except Exception as e:
    return json.dumps({"error": f"Carbon calculation failed: {str(e)}"})


# --- SECURITY & HITL VETO GUARDRAIL TOOLS ---

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
    r"bypass\s+guardrail",
    r"drop\s+table",
    r"delete\s+from",
]


def sanitize_operator_input(text: str) -> str:
  """Deterministic input firewall blocking prompt-injection and SQL payloads."""
  cleaned = str(text or "").strip()[:1000]
  for pat in _INJECTION_PATTERNS:
    if re.search(pat, cleaned, flags=re.IGNORECASE):
      raise HTTPException(
          status_code=400,
          detail="Security Guardrail: Blocked unsafe instruction pattern.",
      )
  return cleaned


def propose_autonomous_intervention(
    shipment_id: str = "ZA-UNASSIGNED",
    corridor: str = "N1 Corridor",
    proposed_action: str = "Divert to nearest cold-storage cross-dock",
    risk_score: float = 75.0,
    **kwargs,
) -> str:
  """Stages a high-risk operational intervention in Supabase with a 60-second HITL veto window."""
  if not supabase:
    return json.dumps({"error": "Supabase client is not connected."})
  try:
    clean_id = (
        re.sub(r"[^A-Za-z0-9\-_]", "", str(shipment_id))[:32] or "ZA-UNKNOWN"
    )
    now_utc = datetime.now(timezone.utc)
    expires_utc = now_utc + timedelta(seconds=60)

    record = {
        "shipment_id": clean_id,
        "corridor": str(corridor)[:120],
        "proposed_action": str(proposed_action)[:300],
        "risk_score": round(float(risk_score), 2),
        "status": "pending",
        "created_at": now_utc.isoformat(),
        "expires_at": expires_utc.isoformat(),
    }
    res = supabase.table("pending_agent_actions").insert(record).execute()
    inserted = (res.data or [record])[0]
    return json.dumps({
        "guardrail_status": "STAGED_FOR_HUMAN_VETO",
        "action_id": inserted.get("id", "pending-id"),
        "shipment_id": clean_id,
        "proposed_action": record["proposed_action"],
        "veto_window_seconds": 60,
        "expires_at": expires_utc.isoformat(),
    })
  except Exception as e:
    return json.dumps({"error": f"Failed to stage intervention: {str(e)}"})


def veto_pending_action(action_id: str = "", **kwargs) -> str:
  """Immediately cancels a pending autonomous intervention by ID."""
  if not supabase:
    return json.dumps({"error": "Supabase client is not connected."})
  try:
    clean_uuid = re.sub(r"[^a-fA-F0-9\-]", "", str(action_id))[:36]
    if not clean_uuid:
      return json.dumps({"error": "Valid action_id UUID is required to veto."})
    res = (
        supabase.table("pending_agent_actions")
        .update({"status": "vetoed"})
        .eq("id", clean_uuid)
        .eq("status", "pending")
        .execute()
    )
    if res.data:
      return json.dumps({"status": "VETOED", "action_id": clean_uuid})
    return json.dumps(
        {"status": "NO_ACTIVE_ACTION_FOUND", "action_id": clean_uuid}
    )
  except Exception as e:
    return json.dumps({"error": f"Veto failed: {str(e)}"})


AVAILABLE_TOOLS = {
    "search_cold_chain_knowledge": search_cold_chain_knowledge,
    "get_active_fleet_telemetry": get_active_fleet_telemetry,
    "get_fleet_aggregate_summary": get_fleet_aggregate_summary,
    "predict_temperature_breach_risk": predict_temperature_breach_risk,
    "get_corridor_live_weather": get_corridor_live_weather,
    "simulate_modal_shift_economics": simulate_modal_shift_economics,
    "calculate_cold_chain_carbon": run_cold_chain_carbon_accounting,
    "carbon_calculator": run_cold_chain_carbon_accounting,
    "calculate_emissions": run_cold_chain_carbon_accounting,
    "propose_autonomous_intervention": propose_autonomous_intervention,
    "veto_pending_action": veto_pending_action,
    "knowledge_search": search_cold_chain_knowledge,
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
            "name": "search_cold_chain_knowledge",
            "description": (
                "Searches the indexed Cold Chain Knowledge Base (South African"
                " Carbon Tax Act 15 of 2019, Climate Change Act 22 of 2024,"
                " GLEC/ISO 14083 carbon accounting, retail logistics reports,"
                " PPECB thermal SOPs, and supply chain engineering research)."
                " REQUIRED whenever asked about regulations, legislation,"
                " cold-chain standards, or academic/industry benchmarks. NEVER"
                " use to compute shipment-level diesel, fuel, CO2e, or g"
                " CO2e/t-km."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Targeted search query combining key concepts and"
                            " exact terms (e.g., 'Carbon Tax Act 15 of 2019"
                            " tax-free allowance schedule')."
                        ),
                    },
                    "category": {
                        "type": ["string", "null"],
                        "enum": [
                            "retail_and_industry_reports",
                            "climate_policy_and_carbon",
                            "thermal_and_sops",
                            "government_and_industry",
                            "meng_coursework_and_assignments",
                            "general",
                            None,
                        ],
                        "description": (
                            "Always pass null to search across all indexed"
                            " documents unless the user explicitly asks only"
                            " for legislation ('climate_policy_and_carbon') or"
                            " PPECB SOPs ('thermal_and_sops')."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                        "description": (
                            "Number of chunks to retrieve (default 2, max 3)."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
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
                " live status, highest temperature excursions, active fleet"
                " spend/cost-to-serve calculations, or checking specific"
                " shipments."
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
    {
        "type": "function",
        "function": {
            "name": "calculate_cold_chain_carbon",
            "description": (
                "MANDATORY deterministic calculator for all cold-chain GHG"
                " emissions (kg CO2e), diesel fuel usage (L), reefer TRU fuel"
                " draw, emission intensity (g CO2e/t-km), and product carbon"
                " footprint (kg CO2e/t cargo). Grounded in Du Plessis, van"
                " Eeden & Goedhals-Gerber (2023) Stellenbosch empirical"
                " coefficients and BS EN 16258 / GLEC Framework v3.1."
                " NEVER compute fuel or emissions manually or via RAG text"
                " search when this tool is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "distance_km": {
                        "type": "number",
                        "description": "Loaded one-way trip distance in km.",
                    },
                    "cargo_weight_tonnes": {
                        "type": "number",
                        "description": "Payload mass in metric tonnes.",
                    },
                    "temperature_regime": {
                        "type": "string",
                        "enum": ["chilled", "frozen", "ambient"],
                        "description": (
                            "'chilled' (>=0°C, 1.5670 L/hr), 'frozen' (-25°C,"
                            " 2.0610 L/hr), or 'ambient' (0 L/hr)."
                        ),
                    },
                    "route_direction": {
                        "type": "string",
                        "enum": ["to_coast", "to_inland", "other"],
                        "description": (
                            "'to_coast' (inland to coastal ports),"
                            " 'to_inland' (coastal climb inland), or 'other'."
                        ),
                    },
                    "trailer_type": {
                        "type": "string",
                        "enum": ["tri_axle_reefer", "interlink_tautliner"],
                        "description": "Vehicle trailer configuration.",
                    },
                    "refrigeration_hours": {
                        "type": ["number", "null"],
                        "description": (
                            "Total active TRU reefer hours. Pass null to"
                            " estimate from distance, speed, and dwell."
                        ),
                    },
                    "empty_distance_km": {
                        "type": ["number", "null"],
                        "description": (
                            "Explicit empty repositioning distance in km. Pass"
                            " null if using empty_running_pct."
                        ),
                    },
                    "empty_running_pct": {
                        "type": "number",
                        "description": (
                            "Empty running percentage (0 to 99, e.g. 20 for"
                            " 20%)."
                        ),
                    },
                    "dwell_and_precool_hours": {
                        "type": "number",
                        "description": (
                            "Extra pre-cooling or terminal dwell hours when"
                            " refrigeration_hours is null (default 0)."
                        ),
                    },
                    "distribution_loss_pct": {
                        "type": "number",
                        "description": (
                            "Post-harvest spoilage/loss percentage for Eq. (6)"
                            " footprint uplift (default 0)."
                        ),
                    },
                },
                "required": [
                    "distance_km",
                    "cargo_weight_tonnes",
                    "temperature_regime",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_autonomous_intervention",
            "description": (
                "REQUIRED whenever a shipment has a temperature breach (>8.0°C)"
                " or CRITICAL breach risk (>=40%). Stages a 60-second"
                " Human-in-the-Loop (HITL) intervention card in the Control"
                " Tower so the operator can veto before execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "shipment_id": {"type": "string"},
                    "corridor": {"type": "string"},
                    "proposed_action": {"type": "string"},
                    "risk_score": {"type": "number"},
                },
                "required": [
                    "shipment_id",
                    "corridor",
                    "proposed_action",
                    "risk_score",
                ],
            },
        },
    },
]

SYSTEM_PROMPT = """
You are an autonomous Cold Chain Operations & MEng Logistics Copilot for Franco.
You assist operations managers with real-time fleet telemetry, predictive risk assessments, weather monitoring, cost-to-serve analytics, carbon calculations, and evidence-backed regulatory/research synthesis.

CRITICAL DIRECTIVE ON KNOWLEDGE RETRIEVAL & PROACTIVE INTENT:
- Never require the user to use rigid keywords, technical syntax, or exact filenames. 
- Whenever the user asks a question about legislation, carbon tax, transport strategies, fleet optimization, supply chain assignments, or academic coursework (even if phrased casually, conversationally, or broadly), you MUST proactively invoke the `search_cold_chain_knowledge` tool.
- Call `search_cold_chain_knowledge` at most ONCE per user turn (with category=null unless strictly legislation or SOPs), then immediately synthesize your response with citations. Do not loop repeatedly with slightly reworded queries.
- If the user's question is conversational or high-level (e.g., "How do in-house fleets compare to 3PLs?" or "What are our carbon tax obligations?"), intelligently reformulate their intent into a targeted conceptual query string for the tool (e.g., query="pure in-house fleet versus third-party logistics cost-to-serve trade-offs").
- Ground your answers strictly in the retrieved excerpts and explicitly cite the source document and page number (e.g., `[Source: 4248323-5act15of2019carbontaxact.pdf, p. 52]`). Never fabricate statutory sections or numbers not present in the retrieved excerpts.

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

CARBON ACCOUNTING & EMISSIONS BENCHMARK:
- NEVER compute diesel consumption, CO2e emissions, or GLEC emission intensities via mental math or RAG text search.
- Whenever asked to calculate trip fuel usage, reefer fuel draw, CO2e emissions, or carbon footprint per tonne/t-km, ALWAYS call `calculate_cold_chain_carbon` first (it contains all built-in Stellenbosch empirical regression coefficients—never ask the user for L/km fuel consumption), then cite the returned Stellenbosch University (Du Plessis, van Eeden & Goedhals-Gerber, 2023) coefficients and BS EN 16258 / GLEC v3.1 metrics.
- Use `simulate_modal_shift_economics` ONLY when specifically comparing road-to-rail modal shift scenarios.

OPERATIONAL COST-TO-SERVE BENCHMARK:
- Formula: Trip Cost (ZAR) = (Distance in km * R18.00) + R200.00 fixed handling fee.
- Always call `get_active_fleet_telemetry(status_filter="active")` to retrieve `distance_km` for cost-to-serve and active fleet spend calculations.
- When answering cost questions, ALWAYS calculate and present:
  1. Total Active Fleet Spend (Sum of all active shipment costs).
  2. Average Cost per Shipment.
  3. The Top 3 Highest-Cost Shipments (with Origin -> Destination, Distance, and ZAR Cost).

RISK CLASSIFICATION BENCHMARK:
- When summarizing output from `predict_temperature_breach_risk`, ALWAYS reflect the exact `risk_classification` and `breach_probability_pct` returned:
  * Breach Probability >= 40.0%: Strictly report as "CRITICAL RISK".
  * Breach Probability 15.0% - 39.99%: Strictly report as "MODERATE RISK".
  * Breach Probability < 15.0%: Strictly report as "LOW RISK".

HUMAN-IN-THE-LOOP (HITL) INTERVENTION GUARDRAIL:
- Whenever an active shipment exceeds 8.0°C OR `predict_temperature_breach_risk` returns CRITICAL RISK (>= 40.0%), ALWAYS call `propose_autonomous_intervention` to stage a 60-second operator veto window before recommending dispatch changes.

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


def _is_carbon_calculation_query(query: str) -> bool:
  """Detects explicit trip fuel/CO2e calculation requests (excluding RAG/legislation/rail queries)."""
  q = query.lower()
  if any(
      excl in q
      for excl in [
          "act",
          "tax",
          "legislation",
          "schedule",
          "allowance",
          "ppecb",
          "sop",
          "modal shift",
          "rail",
          "knowledge base",
          "search",
          "according to",
          "study",
          "paper",
          "document",
          "du plessis",
      ]
  ):
    return False
  has_metric = any(
      m in q
      for m in [
          "diesel",
          "fuel",
          "co2",
          "co2e",
          "emission intensity",
          "carbon footprint",
          "emissions",
      ]
  )
  has_quantitative_trip = any(
      t in q for t in ["km", "tonne", "tonnes", "t·km", "t-km"]
  )
  return has_metric and has_quantitative_trip


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
    role_instructions = (
        "\nUSER ROLE: Executive. Keep answers brief, focus on total ZAR"
        " impact, OTIF, and high-level summaries. Do not list granular shipment"
        " IDs unless asked."
    )
  elif payload.user_role == "Dispatcher":
    role_instructions = (
        "\nUSER ROLE: Floor Dispatcher. Focus on specific shipment IDs, route"
        " names, thermal excursions, and immediate safety interventions."
    )

  dynamic_system_prompt = SYSTEM_PROMPT + role_instructions

  # 2. INITIALIZE MESSAGES
  messages: List[Dict[str, Any]] = [
      {"role": "system", "content": dynamic_system_prompt}
  ]

  if payload.chat_history:
    for msg in payload.chat_history[-6:]:
      if (
          msg.role in ("user", "assistant")
          and msg.content
          and msg.content.strip()
      ):
        messages.append({"role": msg.role, "content": msg.content.strip()})

  clean_query = sanitize_operator_input(payload.user_query)
  messages.append({"role": "user", "content": clean_query})

  force_carbon_turn_1 = _is_carbon_calculation_query(clean_query)
  max_tool_iterations = 3
  iteration = 0
  empty_retried = False

  try:
    while iteration < max_tool_iterations:
      iteration += 1

      turn_tool_choice = (
          {
              "type": "function",
              "function": {"name": "calculate_cold_chain_carbon"},
          }
          if (iteration == 1 and force_carbon_turn_1)
          else "auto"
      )

      # 3. FULL AUTONOMY: tool_choice="auto" paired with 429 failover & expanded token headroom.
      completion = await _create_completion_with_fallback(
          messages=messages,
          tools=GROQ_TOOLS,
          tool_choice=turn_tool_choice,
          reasoning_effort="low",
          temperature=0.1,
          max_tokens=2048,
      )
      response_msg = completion.choices[0].message

      # If no tools were called, check for empty reasoning-channel turn before replying
      if not response_msg.tool_calls:
        content = response_msg.content
        if content and content.strip():
          return AgentQueryResponse(response=content)

        if not empty_retried:
          empty_retried = True
          logger.warning(
              "Model returned empty content and no tool_calls on iteration %d;"
              " retrying once with explicit tool/synthesis nudge.",
              iteration,
          )
          messages.append({
              "role": "system",
              "content": (
                  "Your previous response was empty. Either call the"
                  " appropriate tool to answer the user's question now, or"
                  " provide a direct, complete text answer."
              ),
          })
          continue

        return AgentQueryResponse(
            response=(
                "I am monitoring your active cold-chain fleet. You can request"
                " corridor weather, volume rankings, or cost breakdowns."
            )
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
            tool_output = json.dumps(
                {"error": f"Tool execution error: {str(err)}"}
            )
        else:
          tool_output = json.dumps(
              {"error": f"Function '{func_name}' is not recognized."}
          )

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": func_name,
            "content": str(tool_output),
        })

    # Force synthesis without further tool calls
    messages.append({
        "role": "system",
        "content": (
            "Tool execution budget reached. Do NOT call any further tools."
            " Provide your final answer now using only the retrieved results"
            " above."
        ),
    })
    final_res = await _create_completion_with_fallback(
        messages=messages,
        tools=GROQ_TOOLS,
        tool_choice="none",
        reasoning_effort="low",
        temperature=0.1,
        max_tokens=2048,
    )
    return AgentQueryResponse(
        response=(
            final_res.choices[0].message.content
            or "Completed multi-step fleet telemetry audit."
        )
    )

  except HTTPException:
    raise
  except Exception as e:
    logger.exception("Agent execution failed: %s", str(e))
    raise HTTPException(status_code=500, detail=str(e))


@router.get("/actions/pending")
async def list_pending_actions():
  if not supabase:
    return {"actions": []}
  try:
    now_iso = datetime.now(timezone.utc).isoformat()
    # Auto-transition expired pending actions to 'executed'
    supabase.table("pending_agent_actions").update({"status": "executed"}).eq(
        "status", "pending"
    ).lt("expires_at", now_iso).execute()

    res = (
        supabase.table("pending_agent_actions")
        .select("*")
        .eq("status", "pending")
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    return {"actions": res.data or []}
  except Exception as e:
    logger.warning("Pending actions check failed: %s", e)
    return {"actions": []}


@router.post("/actions/{action_id}/veto")
async def veto_action_endpoint(action_id: str):
  result = json.loads(veto_pending_action(action_id=action_id))
  if "error" in result:
    raise HTTPException(status_code=400, detail=result["error"])
  return result
