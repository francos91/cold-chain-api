import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from supabase import Client, create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logistics-api")

app = FastAPI(title="Logistics Data Pipeline API", version="1.2.0")

# Secure environment initialization
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Critical Error: Supabase environment variables are missing.")

# Connect to the cloud database
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class ShipmentCreate(BaseModel):
    shipment_id: str
    origin: str
    destination: str
    temperature_celsius: float
    status: str
    distance_km: Optional[float] = None
    transit_time_hours: Optional[float] = None


class ShipmentResponse(ShipmentCreate):
    id: int
    logged_at: str


@app.get("/", tags=["System"])
def home():
    return {"status": "online", "message": "Logistics API Pipeline is active."}


@app.get("/health", tags=["System"])
def health():
    """Liveness probe that actually touches Supabase."""
    try:
        supabase.table("cold_chain_tracking").select("id").limit(1).execute()
        return {"status": "ok", "db": "reachable"}
    except Exception:
        logger.exception("health check failed")
        raise HTTPException(status_code=503, detail="Database unreachable")


@app.post(
    "/shipments/",
    response_model=List[ShipmentResponse],
    tags=["Shipments"],
)
def create_shipment(shipment: ShipmentCreate):
    try:
        # exclude_none prevents null values overwriting DB defaults
        payload = shipment.model_dump(exclude_unset=True, exclude_none=True)
        response = supabase.table("cold_chain_tracking").insert(payload).execute()
        return response.data
    except Exception:
        logger.exception("insert failed: %s", shipment.shipment_id)
        raise HTTPException(status_code=500, detail="Insert failed")


@app.post(
    "/shipments/bulk/",
    response_model=List[ShipmentResponse],
    tags=["Shipments"],
)
def create_shipments_bulk(shipments: List[ShipmentCreate]):
    """Bulk insert — used by the synthetic data feeder (4,500 rows)."""
    if not shipments:
        return []
    try:
        payload = [
            s.model_dump(exclude_unset=True, exclude_none=True) for s in shipments
        ]
        response = supabase.table("cold_chain_tracking").insert(payload).execute()
        return response.data
    except Exception:
        logger.exception("bulk insert failed: %d rows", len(shipments))
        raise HTTPException(status_code=500, detail="Bulk insert failed")


@app.get(
    "/shipments/",
    response_model=List[ShipmentResponse],
    tags=["Shipments"],
)
def get_all_shipments(
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    """Paginated read. Supabase caps a single response at 1000 rows, so
    callers must page through with limit/offset to retrieve everything."""
    try:
        response = (
            supabase.table("cold_chain_tracking")
            .select("*")
            .range(offset, offset + limit - 1)
            .execute()
        )
        return response.data
    except Exception:
        logger.exception("fetch failed (offset=%d, limit=%d)", offset, limit)
        raise HTTPException(status_code=500, detail="Fetch failed")
