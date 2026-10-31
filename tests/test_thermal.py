import pytest

def check_temperature_excursion(temp_celsius, threshold_max=8.0):
    """Returns True if temperature breaches safe cold chain threshold."""
    return temp_celsius > threshold_max

def test_thermal_excursion_detection():
    # Safe temperature for chilled fish
    assert check_temperature_excursion(4.2) == False
    
    # Borderline safe temperature
    assert check_temperature_excursion(8.0) == False
    
    # Excursion temperature (spoilage risk)
    assert check_temperature_excursion(9.5) == True
    assert check_temperature_excursion(14.1) == True
