from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from supabase import create_client, Client

app = FastAPI(title="Logistics Data Pipeline API", version="1.1.0")

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

@app.post("/shipments/", response_model=List[ShipmentResponse], tags=["Shipments"])
def create_shipment(shipment: ShipmentCreate):
    try:
        # Convert Pydantic model directly to dictionary payload
        payload = shipment.model_dump(exclude_unset=True)
        response = supabase.table("cold_chain_tracking").insert(payload).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/shipments/", response_model=List[ShipmentResponse], tags=["Shipments"])
def get_all_shipments():
    try:
        response = supabase.table("cold_chain_tracking").select("*").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
