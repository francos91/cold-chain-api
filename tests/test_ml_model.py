import pytest
import numpy as np

def mock_model_predict(features):
    """Simulates the 5-feature Random Forest risk prediction inference."""
    # Features: [distance, ambient_temp, duration, historical_delay, route_risk_score]
    if len(features) != 5:
        raise ValueError("Model expects exactly 5 features")
    # Returns a risk probability between 0 and 1
    return 0.85 if features[1] > 30.0 else 0.15

def test_ml_feature_validation():
    # Test valid 5-feature input shape
    valid_features = [450.0, 32.5, 6.0, 1.2, 0.4]
    risk_score = mock_model_predict(valid_features)
    
    assert 0.0 <= risk_score <= 1.0
    assert risk_score == 0.85  # High ambient temp triggers higher risk prediction

def test_ml_invalid_shape_raises_error():
    invalid_features = [450.0, 32.5]  # Only 2 features instead of 5
    with pytest.raises(ValueError):
        mock_model_predict(invalid_features)
