import sys
import os
from fastapi.testclient import TestClient

# Add the Backend folder to Python's path so TestClient can import app from main.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../Backend")))

from main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the base system status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Logistics API Pipeline is active" in data["message"]

def test_health_check_endpoint():
    """Test the database liveness probe."""
    response = client.get("/health")
    # Will return 200 if Supabase is reachable, or 503 if offline/credentials are missing in local test env
    assert response.status_code in [200, 503]

def test_simulate_intermodal_endpoint():
    """Test the road-to-rail modal shift calculation microservice route."""
    payload = {
        "corridor_key": "Johannesburg_to_Durban (Natcor)",
        "shift_percentage": 30.0,
        "total_tkm": 500000.0,
        "total_shipments": 40
    }
    response = client.post("/api/simulate-intermodal", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Verify expected calculation keys returned by the microservice
    assert "carbon_savings_tonnes" in data
    assert "carbon_tax_savings_zar" in data
    assert "new_cost" in data
    assert data["carbon_savings_tonnes"] > 0
