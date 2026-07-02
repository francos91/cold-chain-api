from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import sys
from supabase import create_client, Client

app = FastAPI(title="Logistics Data Pipeline API")

# --- DEBUG BLOCK: Force Render to tell us what it sees ---
print("🚨 DEBUG: Checking Environment Variables...")
env_keys = list(os.environ.keys())

if "SUPABASE_URL" in env_keys:
    print("✅ SUPABASE_URL is present!")
else:
    print("❌ SUPABASE_URL is completely missing from Render!")
    
if "SUPABASE_KEY" in env_keys:
    print("✅ SUPABASE_KEY is present!")
else:
    print("❌ SUPABASE_KEY is completely missing from Render!")
sys.stdout.flush()
# ---------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Critical Error: Supabase environment variables are missing!")

# Connect to the cloud database
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class ShipmentCreate(BaseModel):
    shipment_id: str
    origin: str
    destination: str
    temperature_celsius: float
    status: str

class ShipmentResponse(ShipmentCreate):
    id: int
    logged_at: str

@app.get("/")
def home():
    return {"status": "online", "message": "Logistics API Pipeline is active."}

@app.post("/shipments/", response_model=List[ShipmentResponse])
def create_shipment(shipment: ShipmentCreate):
    try:
        payload = {
            "shipment_id": shipment.shipment_id,
            "origin": shipment.origin,
            "destination": shipment.destination,
            "temperature_celsius": shipment.temperature_celsius,
            "status": shipment.status
        }
        response = supabase.table("cold_chain_tracking").insert(payload).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/shipments/", response_model=List[ShipmentResponse])
def get_all_shipments():
    try:
        response = supabase.table("cold_chain_tracking").select("*").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
