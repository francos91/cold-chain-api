import os
import sys

# 1. Set mock environment variables BEFORE importing main.py so validation passes
os.environ.setdefault("SUPABASE_URL", "https://mock-supabase-url.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "mock-supabase-key-for-testing")

# 2. Add the Backend folder to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../Backend")))

from fastapi.testclient import TestClient
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
    # With mock credentials, Supabase will be unreachable, returning 503 (which proves the liveness probe works)
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
