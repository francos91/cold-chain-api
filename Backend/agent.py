import json
import logging
import os
import re
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from groq import AsyncGroq
import joblib
import pandas as pd
from pydantic import BaseModel
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


# --- 2. EXECUTABLE TOOLS ---


def get_active_fleet_telemetry(only_breaches: bool = False, **kwargs) -> str:
  """Fetches live telemetry for active shipments from Supabase."""
  if not supabase:
    return json.dumps({"error": "Supabase client is not connected."})
  try:
    query = (
        supabase.table("cold_chain_tracking")
        .select(
            "shipment_id,origin,destination,temperature_celsius,status,distance_km,transit_time_hours,weight_tonnes"
        )
        .not_.eq("status", "delivered")
    )
    if only_breaches:
      query = query.gt("temperature_celsius", 8.0)

    # Capped at 15 to stay strictly within token budgets and avoid read timeouts
    res = query.order("temperature_celsius", desc=True).limit(15).execute()
    return json.dumps(res.data or [])
  except Exception as e:
    return json.dumps({"error": f"Database query failed: {str(e)}"})


def predict_temperature_breach_risk(
    origin: str = "Johannesburg",
    destination: str = "Cape Town",
    is_delayed: bool = False,
    transit_time_hours: float = 24.0,
    distance_km: float = 600.0,
    **kwargs,
) -> str:
  """Runs the containerized Random Forest model to predict continuous probability of a temperature breach (>8°C)."""
  if rf_model is None or route_encoder is None:
    return json.dumps(
        {"error": "ML models not available in backend container."}
    )

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
      "transit_time_hours": float(transit_time_hours),
  }])

  prob = float(rf_model.predict_proba(features)[0, 1])
  return json.dumps({
      "route": route_key,
      "breach_probability_pct": round(prob * 100, 2),
      "risk_classification": (
          "CRITICAL RISK"
          if prob > 0.40
          else "MODERATE RISK"
          if prob > 0.15
          else "LOW RISK"
      ),
      "parameters": {
          "is_delayed": is_delayed,
          "transit_time_hours": transit_time_hours,
          "distance_km": distance_km,
      },
  })


def get_corridor_live_weather(
    location_name: str = "Beaufort West", **kwargs
) -> str:
  """Fetches real-time ambient weather from Open-Meteo."""
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
  lat, lon = COORDINATES.get(loc_clean, (-32.3567, 22.5830))
  url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure"

  try:
    r = requests.get(url, timeout=6)
    r.raise_for_status()
    current = r.json().get("current", {})
    return json.dumps({
        "waypoint": location_name,
        "ambient_temp_celsius": current.get("temperature_2m"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "surface_pressure_hpa": current.get("surface_pressure"),
        "highveld_thermal_penalty": (
            "21% altitude cooling efficiency loss applied"
            if current.get("surface_pressure", 1013) < 850
            else "Nominal cooling"
        ),
    })
  except Exception as e:
    return json.dumps({"error": f"Weather fetch failed: {str(e)}"})


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
    "predict_temperature_breach_risk": predict_temperature_breach_risk,
    "get_corridor_live_weather": get_corridor_live_weather,
    "simulate_modal_shift_economics": simulate_modal_shift_economics,
    "weather_monitoring": get_corridor_live_weather,
    "get_weather": get_corridor_live_weather,
    "predict_risk": predict_temperature_breach_risk,
    "fleet_telemetry": get_active_fleet_telemetry,
}

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_active_fleet_telemetry",
            "description": (
                "REQUIRED for active fleet status, summaries, temperature"
                " excursions, operational costs, financial exposure, or"
                " cost-to-serve calculations. Fetches distances and"
                " temperatures from Supabase."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "only_breaches": {
                        "type": "boolean",
                        "description": (
                            "Set to True to filter exclusively for shipments"
                            " above 8.0°C."
                        ),
                    }
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
                    "distance_km": {"type": "number"},
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
            "name": "get_corridor_live_weather",
            "description": (
                "Fetches ambient weather and altitude pressure for key"
                " corridors (e.g., Beaufort West, Johannesburg, Durban)."
            ),
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

OPERATIONAL COST-TO-SERVE BENCHMARK:
- Formula: Trip Cost (ZAR) = (Distance in km * R18.00) + R200.00 fixed handling fee.
- When answering cost questions, ALWAYS calculate and present:
  1. Total Active Fleet Spend (Sum of all active shipment costs).
  2. Average Cost per Shipment.
  3. The Top 3 Highest-Cost Shipments (with Origin -> Destination, Distance, and ZAR Cost).

RISK CLASSIFICATION BENCHMARK:
- When summarizing output from `predict_temperature_breach_risk`, NEVER contradict the returned `risk_classification`:
  * Breach Probability > 40%: Must strictly be reported as "CRITICAL RISK".
  * Breach Probability 15% - 40%: Must strictly be reported as "MODERATE RISK".
  * Breach Probability < 15%: Must strictly be reported as "LOW RISK".
- If the tool outputs a probability of 67%, declare it as "CRITICAL RISK" and advise on immediate refrigeration or routing interventions.

OPERATIONAL BENCHMARKS:
- "Poorest / Worst Performer" = Shipment with highest current temperature, breach status (>8.0°C), or status == 'delayed'.
- "Best Performer" = Shipment that is 'in-transit', non-delayed, and closest to ideal cold chain target (0.5°C to 4.5°C).
- "Summary / Overall" = Report active fleet count, delayed vs in-transit count, and detail any active excursions (>8.0°C).

CURRENCY & UNIT FIDELITY:
- Always format currency as 'R' or 'ZAR' (e.g., R29,102.60).
- Never round single Rand values into 'k' (e.g., R256.20 must never be written as R256k).
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
        detail=(
            "Groq Client not initialized. Set GROQ_API_KEY in Render"
            " environment."
        ),
    )

  user_text = payload.user_query.strip().lower()

  # DETERMINISTIC INTENT DETECTION:
  # If the user asks for costs, fleet summaries, or active loads, force get_active_fleet_telemetry
  needs_telemetry = bool(
      re.search(
          r"\b(cost|costs|spend|financial|summary|fleet|active|operations|performer|pick)\b",
          user_text,
      )
  )

  messages = [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": payload.user_query},
  ]

  max_tool_iterations = 3
  iteration = 0

  try:
    while iteration < max_tool_iterations:
      iteration += 1

      # Force telemetry tool calling on iteration 1 if the query matches operational intent
      if iteration == 1 and needs_telemetry:
        tool_choice_param = {
            "type": "function",
            "function": {"name": "get_active_fleet_telemetry"},
        }
      else:
        tool_choice_param = "auto"

      completion = await groq_client.chat.completions.create(
          model=ACTIVE_MODEL,
          messages=messages,
          tools=GROQ_TOOLS,
          tool_choice=tool_choice_param,
          temperature=0.1,
          max_tokens=900,
      )
      response_msg = completion.choices[0].message

      # If no tools were called
      if not response_msg.tool_calls:
        content = response_msg.content
        if content and content.strip():
          return AgentQueryResponse(response=content)

        # Deterministic Python fallback for cost calculations if LLM returned empty content
        if needs_telemetry:
          raw_data = json.loads(get_active_fleet_telemetry())
          if isinstance(raw_data, list) and len(raw_data) > 0:
            total_cost = sum(
                (r.get("distance_km", 0) * 18.0) + 200.0 for r in raw_data
            )
            avg_cost = total_cost / len(raw_data)
            top_shipments = sorted(
                raw_data,
                key=lambda x: (x.get("distance_km", 0) * 18.0) + 200.0,
                reverse=True,
            )[:3]

            breakdown = "\n".join([
                f"- **{s.get('shipment_id')}** ({s.get('origin')} ->"
                f" {s.get('destination')}): {s.get('distance_km', 0):,.1f} km"
                f" = **R{((s.get('distance_km', 0) * 18.0) + 200.0):,.2f}**"
                for s in top_shipments
            ])

            return AgentQueryResponse(
                response=(
                    "### Operational Cost-to-Serve Summary\n\n"
                    f"- **Active Shipments Analyzed:** {len(raw_data)}\n"
                    f"- **Total Active Fleet Spend:** **R{total_cost:,.2f}**\n"
                    f"- **Average Cost per Shipment:** **R{avg_cost:,.2f}**\n\n"
                    f"**Top 3 Highest-Cost Shipments:**\n{breakdown}"
                )
            )

        return AgentQueryResponse(
            response=(
                "I am monitoring your active cold-chain fleet. You can request"
                " cost breakdowns, breach risk predictions, corridor weather,"
                " or a fleet summary."
            )
        )

      # Build clean assistant payload dictionary (strips internal Groq annotations)
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

      # Execute each tool call emitted in this turn
      for tool_call in response_msg.tool_calls:
        func_name = tool_call.function.name

        try:
          func_args = json.loads(tool_call.function.arguments or "{}")
        except Exception:
          func_args = {}

        tool_func = AVAILABLE_TOOLS.get(func_name)

        if tool_func:
          logger.info(
              "Iteration %d | Executing tool: %s with args: %s",
              iteration,
              func_name,
              func_args,
          )
          try:
            tool_output = tool_func(**func_args)
          except Exception as err:
            tool_output = json.dumps(
                {"error": f"Tool execution error: {str(err)}"}
            )
        else:
          logger.warning("Unknown tool attempted: %s", func_name)
          tool_output = json.dumps({
              "error": (
                  f"Function '{func_name}' is not recognized. Valid tools:"
                  f" {list(AVAILABLE_TOOLS.keys())}"
              )
          })

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": func_name,
            "content": str(tool_output),
        })

    # Final synthesis call
    final_res = await groq_client.chat.completions.create(
        model=ACTIVE_MODEL, messages=messages, temperature=0.1, max_tokens=1000
    )
    return AgentQueryResponse(
        response=final_res.choices[0].message.content
        or "Completed multi-step fleet telemetry audit."
    )

  except Exception as e:
    logger.exception("Agent execution failed: %s", str(e))
    raise HTTPException(status_code=500, detail=str(e))
