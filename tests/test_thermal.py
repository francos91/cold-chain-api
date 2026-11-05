import pytest

def check_temperature_excursion(temp_celsius: float, threshold_max: float = 8.0) -> bool:
    """Returns True if the temperature breaches the critical cold chain threshold."""
    return temp_celsius > threshold_max

def test_thermal_excursion_safe_range():
    """Test temperatures well within safe operating limits for chilled perishables."""
    assert check_temperature_excursion(4.2) is False
    assert check_temperature_excursion(0.0) is False

def test_thermal_excursion_boundary():
    """Test exact boundary condition at the 8.0°C limit (non-breach)."""
    assert check_temperature_excursion(8.0) is False

def test_thermal_excursion_breaches():
    """Test temperatures exceeding the maximum threshold, triggering excursion and spoilage risk flags."""
    assert check_temperature_excursion(8.1) is True
    assert check_temperature_excursion(9.5) is True
    assert check_temperature_excursion(14.1) is True
