import os
import joblib
import numpy as np
import pytest

# Locate model artifacts inside the dashboard folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "dashboard", "advanced_model.pkl")
ENCODER_PATH = os.path.join(ROOT_DIR, "dashboard", "le_route.pkl")

def test_model_artifacts_exist():
    """Ensure trained Scikit-learn model and encoder files exist in the dashboard directory."""
    assert os.path.exists(MODEL_PATH), f"CRITICAL: Model file missing at {MODEL_PATH}"
    assert os.path.exists(ENCODER_PATH), f"CRITICAL: Encoder file missing at {ENCODER_PATH}"

def test_model_inference_probabilities():
    """Test that the real advanced_model.pkl successfully executes predict_proba and returns valid bounds."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        pytest.skip("Model artifacts not available in this test environment.")

    # Load model artifacts using joblib from dashboard folder
    model = joblib.load(MODEL_PATH)
    
    # The model expects 5 features: ["route_enc", "route_risk_score", "is_delayed", "distance_km", "transit_time_hours"]
    sample_features = np.array([[0, 0.05, 0, 568.0, 9.5]])
    
    # Run inference
    probabilities = model.predict_proba(sample_features)
    
    # Validate shape and bounds
    assert probabilities.shape == (1, 2), "Model output must be a binary classification probability array of shape (1, 2)."
    
    risk_probability = probabilities[0, 1]
    assert 0.0 <= risk_probability <= 1.0, f"Predicted risk probability {risk_probability} is outside valid [0.0, 1.0] range."
