import pytest

def mock_model_predict(ambient_temp):
    """Simulates a lightweight risk prediction based on ambient temperature."""
    if ambient_temp > 30.0:
        return 0.85  # High risk
    return 0.15      # Low risk

def test_ml_feature_validation():
    # Test high temperature risk trigger
    risk_score_hot = mock_model_predict(32.5)
    assert 0.0 <= risk_score_hot <= 1.0
    assert risk_score_hot == 0.85

def test_ml_low_risk_validation():
    # Test normal temperature safe state
    risk_score_cool = mock_model_predict(18.0)
    assert risk_score_cool == 0.15
