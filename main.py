from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from supabase import create_client, Client

# Initialize the FastAPI app engine
app = FastAPI(title="Logistics Data Pipeline API")

# Secure, fail-fast environment variables for production
# This ensures no secret keys are ever exposed in your code
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Crash the app immediately if keys are missing, preventing silent failures
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Critical Error: Supabase environment variables are missing!")

# Connect to the cloud database
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Define the expected structure of incoming JSON data (Data Validation)
class ShipmentCreate(BaseModel):
    shipment_id: str
    origin: str
    destination: str
    temperature_celsius: float
    status: str

# Define the structure of data returning from the database
class ShipmentResponse(ShipmentCreate):
    id: int
    logged_at: str

@app.get("/")
def home():
    return {"status": "online", "message": "Logistics API Pipeline is active."}

# Endpoint 1: Ingest Data (POST)
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
        # Insert data into our cloud PostgreSQL table via the Supabase client
        response = supabase.table("cold_chain_tracking").insert(payload).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: Extract Data (GET)
@app.get("/shipments/", response_model=List[ShipmentResponse])
def get_all_shipments():
    try:
        # Query all records from our cloud PostgreSQL table
        response = supabase.table("cold_chain_tracking").select("*").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
