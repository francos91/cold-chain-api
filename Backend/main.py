import logging
import os
from typing import List, Optional

import joblib
import pandas as pd
import numpy as np
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

# Load ML Models for Inference Engine
try:
    model = joblib.load("advanced_model.pkl")
    le_route = joblib.load("le_route.pkl")
    logger.info("ML Models loaded successfully.")
except Exception as e:
    logger.warning("ML models not found or failed to load. Ensure they are copied in the Dockerfile.")
    model, le_route = None, None


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


@app.get("/api/dashboard", tags=["Dashboard"])
def get_dashboard_data():
    """Fetches raw data, runs ML predictions, and returns ready-to-use JSON."""
    try:
        # 1. Fetch raw data from Supabase
        response = supabase.table("cold_chain_tracking").select("*").execute()
        if not response.data:
            return {"raw": [], "features": []}
        
        df_feat = pd.DataFrame(response.data)
        
        # 2. Feature Engineering
        df_feat["route"] = df_feat["origin"] + "_to_" + df_feat["destination"]
        risk_map = df_feat.groupby("route")["temperature_celsius"].apply(lambda x: (x > 8.0).mean()).to_dict()
        df_feat["route_risk_score"] = df_feat["route"].map(risk_map)
        
        if le_route is not None:
            try:
                df_feat["route_enc"] = le_route.transform(df_feat["route"])
            except ValueError:
                df_feat["route_enc"] = 0
        else:
            df_feat["route_enc"] = 0
            
        df_feat["is_delayed"] = (df_feat["status"] == "delayed").astype(int)
        df_feat["is_risk"] = (df_feat["temperature_celsius"] > 8.0).astype(int)
        
        df_feat["distance_km"] = pd.to_numeric(df_feat.get("distance_km", 500), errors="coerce").fillna(500)
        df_feat["weight_tonnes"] = pd.to_numeric(df_feat.get("weight_tonnes", 20.0), errors="coerce").fillna(20.0)
        
        if "transit_time_hours" not in df_feat.columns or df_feat["transit_time_hours"].isnull().all():
            base_transit = df_feat["distance_km"] / 65.0
            df_feat["transit_time_hours"] = base_transit + (df_feat["is_delayed"] * 8.0)
        else:
            df_feat["transit_time_hours"] = pd.to_numeric(df_feat["transit_time_hours"], errors="coerce")

        df_feat["is_ontime"] = (df_feat["status"] != "delayed").astype(int)
        df_feat["is_compliant"] = (df_feat["is_risk"] == 0).astype(int)
        df_feat["is_perfect"] = ((df_feat["is_ontime"] == 1) & (df_feat["is_compliant"] == 1)).astype(int)
        df_feat["cost_per_shipment"] = (df_feat["distance_km"] * 18) + 200

        # 3. ML Inference
        X_pred = df_feat[["route_enc", "route_risk_score", "is_delayed", "distance_km", "transit_time_hours"]]
        if not X_pred.empty and model is not None:
            df_feat["risk_probability"] = model.predict_proba(X_pred)[:, 1]
        else:
            df_feat["risk_probability"] = 0.0

        # Return as JSON (convert NaN/Infinity values to handle JSON serialization properly)
        df_feat = df_feat.fillna(0)
        
        return {
            "raw": response.data,
            "features": df_feat.to_dict(orient="records")
        }
    except Exception as e:
        logger.exception("Failed to generate dashboard data payload")
        raise HTTPException(status_code=500, detail=str(e))
