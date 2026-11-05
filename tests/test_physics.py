import pytest

def calculate_emissions(tkm: float, factor: float) -> float:
    """Calculates total carbon emissions in tonnes from tonne-kilometers and emission factor (g CO2e/tkm)."""
    return (tkm * factor) / 1_000_000

def calculate_carbon_tax(emissions_tonnes: float, tax_rate_zar: float) -> float:
    """Calculates total carbon tax liability in ZAR based on net emissions and effective tax rate."""
    return emissions_tonnes * tax_rate_zar

def calculate_shipment_cost(distance_km: float, rate_per_km: float, fixed_cost: float) -> float:
    """Calculates total cost-to-serve per shipment based on distance physics and fixed overhead."""
    return (distance_km * rate_per_km) + fixed_cost

def test_iso_14083_carbon_accounting():
    """Test diesel baseline emission factor calculation (73.2 g CO2e/tkm)."""
    tkm = 100_000  # 100,000 tonne-kilometers
    emissions = calculate_emissions(tkm, 73.2)
    assert emissions == 7.32, f"Expected 7.32 tonnes, got {emissions}"

def test_carbon_tax_liability():
    """Test South African effective carbon tax rate of R28.00/tonne (incorporating basic allowances)."""
    emissions_tonnes = 10.0
    tax_rate = 28.0
    tax_liability = calculate_carbon_tax(emissions_tonnes, tax_rate)
    assert tax_liability == 280.0, f"Expected R280, got R{tax_liability}"

def test_cost_to_serve_formula():
    """Test transport cost formula: (distance * rate) + fixed overhead."""
    distance_km = 500.0
    rate_per_km = 18.0
    fixed_cost = 200.0
    total_cost = calculate_shipment_cost(distance_km, rate_per_km, fixed_cost)
    assert total_cost == 9200.0, f"Expected R9,200, got R{total_cost}"
