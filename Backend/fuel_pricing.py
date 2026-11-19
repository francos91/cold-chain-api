from datetime import datetime, timezone
from duckduckgo_search import DDGS


def get_sa_diesel_rate(
    supabase, region: str = "inland", grade: str = "50ppm", retail: bool = False
) -> dict:
  """Reads active SA DMPR diesel price (ZAR/L) from Supabase system_config, with monthly DDGS fallback."""
  default_rates = {
      "effective_date": "2026-11-04",
      "diesel_50ppm_inland_zar": 31.60,
      "diesel_50ppm_coastal_zar": 30.73,
      "diesel_500ppm_inland_zar": 30.86,
      "diesel_500ppm_coastal_zar": 29.98,
      "retail_pump_margin_zar": 1.80,
      "source_note": (
          "Official SA DMPR wholesale schedule (4 Nov 2026). Add ~R1.80/L for"
          " retail pump price."
      ),
  }

  now = datetime.now(timezone.utc)
  cached = default_rates.copy()
  source_status = "fallback_default"
  is_stale = True

  if supabase:
    try:
      res = (
          supabase.table("system_config")
          .select("value, updated_at")
          .eq("key", "sa_diesel_prices")
          .execute()
      )
      if res.data:
        cached = res.data[0]["value"]
        updated_at = datetime.fromisoformat(
            res.data[0]["updated_at"].replace("Z", "+00:00")
        )
        if updated_at.year == now.year and updated_at.month == now.month:
          is_stale = False
          source_status = "supabase_dmpr_cache"
    except Exception:
      pass

  zone = "coastal" if str(region).lower().strip() == "coastal" else "inland"
  fuel_grade = "500ppm" if "500" in str(grade) else "50ppm"
  rate_key = f"diesel_{fuel_grade}_{zone}_zar"
  zar_per_litre = float(cached.get(rate_key, 31.60))

  if retail:
    zar_per_litre += float(cached.get("retail_pump_margin_zar", 1.80))

  result = {
      "status": source_status,
      "region": zone,
      "grade": fuel_grade,
      "pricing_tier": "retail_pump_est" if retail else "dmpr_wholesale",
      "active_zar_per_litre": round(zar_per_litre, 2),
      **cached,
  }

  if is_stale:
    try:
      month_str = now.strftime("%B %Y")
      with DDGS() as ddgs:
        hits = list(
            ddgs.text(
                f"South Africa official diesel price 50ppm wholesale {month_str}",
                max_results=3,
            )
        )
        result["status"] = "live_ddgs_supplemented"
        result["live_dmpr_snippets"] = " | ".join([h["body"] for h in hits])[
            :650
        ]
    except Exception:
      pass

  return result
