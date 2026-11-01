import pytest

def calculate_emissions(tkm, factor):
    return (tkm * factor) / 1_000_000

def test_iso_14083_carbon_accounting():
    # Test diesel baseline emission factor calculation (73.2 g CO2e/tkm)
    tkm = 100_000 # 100,000 tonne-kilometers
    emissions = calculate_emissions(tkm, 73.2)
    assert emissions == 7.32, f"Expected 7.32 tonnes, got {emissions}"

def test_carbon_tax_liability():
    # Test South African effective carbon tax rate of R28.00/tonne (incorporating basic allowances)
    emissions_tonnes = 10.0
    tax_rate = 28.0
    tax_liability = emissions_tonnes * tax_rate
    assert tax_liability == 280.0, f"Expected R280, got R{tax_liability}"

def test_cost_to_serve_formula():
    # Test transport cost formula: (distance * rate) + fixed
    distance_km = 500
    rate_per_km = 18.0
    fixed_cost = 200.0
    total_cost = (distance_km * rate_per_km) + fixed_cost
    assert total_cost == 9200.0, f"Expected R9,200, got R{total_cost}"
