from typing import Literal, Optional, Dict, Any

# ==============================================================================
# EMPIRICAL CONSTANTS: STELLENBOSCH UNIVERSITY & GLEC FRAMEWORK v3.1 / ISO 14083
# ==============================================================================

# Du Plessis et al. (2023) - "Calculating Fuel Usage and Emissions for Refrigerated
# Road Transport Using Real-World Data", Transportation Research Part D 117, 103623
# Table 2 (p. 13): Empirical Fleet Regression Coefficients
CORRIDOR_COEFFICIENTS = {
    "tri_axle_reefer": {
        "to_coast": {
            "A_empty_l_per_km": 0.3744,      # Empty tri-axle semi-reefer consumption (L/km)
            "B_load_l_per_tkm": 0.001872,    # Payload mass fuel penalty (L/tonne-km)
            "description": "Tri-axle semi-reefer: Inland-to-Coast descent profile (e.g., Gauteng to Cape Town/Durban)"
        },
        "to_inland": {
            "A_empty_l_per_km": 0.3759,      # Empty tri-axle semi-reefer consumption (L/km)
            "B_load_l_per_tkm": 0.002482,    # Escarpment climb fuel penalty (L/tonne-km)
            "description": "Tri-axle semi-reefer: Coast-to-Inland ascent profile (N1/N3 escarpment)"
        },
        "other": {
            "A_empty_l_per_km": 0.3809,      # Other routes (Table 2)
            "B_load_l_per_tkm": 0.004282,
            "description": "Tri-axle semi-reefer: Other / regional routes (Du Plessis et al., 2023, Table 2)"
        }
    },
    "interlink_tautliner": {
        "to_coast": {
            "A_empty_l_per_km": 0.5489,      # Empty tandem-tandem interlink tautliner (L/km)
            "B_load_l_per_tkm": 0.006195,
            "description": "Interlink tautliner (ambient only): Inland-to-Coast profile"
        },
        "to_inland": {
            "A_empty_l_per_km": 0.5228,
            "B_load_l_per_tkm": 0.004497,
            "description": "Interlink tautliner (ambient only): Coast-to-Inland profile"
        },
        "other": {
            "A_empty_l_per_km": 0.4991,
            "B_load_l_per_tkm": 0.002629,
            "description": "Interlink tautliner (ambient only): Other / regional routes"
        }
    }
}

# Du Plessis et al. (2023), Table 2 (p. 13) - Empirical TRU (Reefer) Hourly Diesel Burn Rates
REEFER_BURN_RATES_L_PER_HR = {
    "frozen": 2.0610,    # Deep-frozen setpoint (-25°C; mean 2.0610 L/hr, SD 0.5065 L/hr)
    "chilled": 1.5670,   # Chilled/controlled setpoint (>= 0°C; mean 1.5670 L/hr, SD 0.4285 L/hr)
    "ambient": 0.00      # Dry van / non-refrigerated haul
}

# BS EN 16258:2012 Well-to-Wheel (WTW) Diesel Factor (Du Plessis et al., 2023, Eq. 2, p. 14)
DIESEL_WTW_KG_CO2E_PER_LITRE = 3.24      # 3.24 kg CO2e/L WTW (BSI, 2012; GLEC aligned)

# GLEC Framework v3.1 (Table 9 & Slide 33) - Activity-Based EIFs & Regional Uplifts
GLEC_V3_1_ROAD_WTW_G_PER_TKM = {
    "artic_40t_heavy": 80.0,
    "artic_40_44t_light": 143.0,
    "artic_40_44t_average": 89.0,
    "artic_40_44t_heavy": 77.0,
    "artic_40_44t_container": 90.0,
    "artic_60t_average": 70.0,
    "artic_60t_heavy": 60.0,
    "artic_60t_container": 70.0,
    "artic_72t_heavy": 47.0,
    "artic_72t_container": 54.0,
}
GLEC_AFRICA_AMBIENT_MULTIPLIER = 1.22           # +22% Africa/Asia regional adjustment (Slide 33)
GLEC_AFRICA_REEFER_MULTIPLIER = 1.22 * 1.15     # +22% regional, then +15% refrigeration (Slide 33)

GLEC_V3_1_HUBS = {
    "transshipment_ambient_kg_per_t": 1.3,
    "transshipment_mixed_kg_per_t": 2.5,
    "storage_transshipment_ambient_kg_per_t": 5.6,
    "storage_transshipment_mixed_kg_per_t": 18.4,
    "warehouse_ambient_kg_per_t": 45.5,
    "warehouse_mixed_kg_per_t": 50.0,
    "maritime_terminal_ambient_kg_per_container": 10.7,
    "maritime_terminal_reefer_kg_per_container": 12.6,
}

GLEC_V3_1_SEA_WTW_G_PER_TEUKM = {
    "europe_africa_dry": 102.0,
    "europe_africa_reefer": 187.0,
    "industry_avg_dry": 72.7,
    "industry_avg_reefer": 148.0,
}
GLEC_SEA_DETOUR_FACTOR = 1.05                   # +5% sea distance uplift (Slide 39)
SA_GRID_EMISSION_FACTOR_KG_PER_KWH = 0.98       # SA Grid Electricity = 0.98 kg CO2e/kWh (Slide 18)


def calculate_cold_chain_carbon(
    distance_km: float,
    cargo_weight_tonnes: float,
    temperature_regime: Literal["chilled", "frozen", "ambient"] = "chilled",
    refrigeration_hours: Optional[float] = None,
    route_direction: Literal["to_coast", "to_inland", "other", "flat_regional"] = "to_coast",
    trailer_type: Literal["tri_axle_reefer", "interlink_tautliner"] = "tri_axle_reefer",
    empty_distance_km: Optional[float] = None,
    empty_running_pct: float = 0.0,
    average_speed_kmh: float = 65.0,
    dwell_and_precool_hours: float = 0.0,
    distribution_loss_pct: float = 0.0
) -> Dict[str, Any]:
    """
    Deterministic carbon accounting and fuel-burn calculator grounded strictly in
    Prof. Leila Goedhals-Gerber & Dr. Martin du Plessis (Stellenbosch University, 2023/2024)
    empirical fleet models and BS EN 16258 / GLEC Framework v3.1 methodology.
    """
    if distance_km <= 0 or cargo_weight_tonnes <= 0:
        raise ValueError("distance_km and cargo_weight_tonnes must be strictly positive.")

    # Map legacy alias 'flat_regional' to Table 2's verified 'other' category
    normalized_direction = "other" if route_direction == "flat_regional" else route_direction
    regime = temperature_regime.lower()

    if trailer_type == "interlink_tautliner" and regime != "ambient":
        raise ValueError("Interlink tautliners only transport ambient cargo in Du Plessis et al. (2023).")

    # 1. Select verified Table 2 empirical corridor coefficients
    trailer_coeffs = CORRIDOR_COEFFICIENTS.get(trailer_type, CORRIDOR_COEFFICIENTS["tri_axle_reefer"])
    coeffs = trailer_coeffs.get(normalized_direction, trailer_coeffs["to_coast"])
    coeff_a = coeffs["A_empty_l_per_km"]
    coeff_b = coeffs["B_load_l_per_tkm"]

    # 2. Select verified Table 2 empirical TRU reefer burn rate (L/hr)
    coeff_c = 0.0 if trailer_type == "interlink_tautliner" else REEFER_BURN_RATES_L_PER_HR.get(regime, 1.5670)

    # 3. Calculate D_Total per Du Plessis et al. (2023) Eq. (A.5): Empty_running = D_empty / D_total
    if empty_distance_km is not None:
        d_empty = max(0.0, empty_distance_km)
        d_total = distance_km + d_empty
        empty_ratio = d_empty / d_total if d_total > 0 else 0.0
    else:
        empty_ratio = max(0.0, empty_running_pct / 100.0 if empty_running_pct > 1.0 else empty_running_pct)
        if empty_ratio >= 1.0:
            raise ValueError("empty_running_pct must be strictly less than 100%.")
        d_total = distance_km / (1.0 - empty_ratio)
        d_empty = d_total - distance_km

    # 4. Refrigeration operating hours (T_ref)
    if coeff_c == 0.0:
        refrigeration_hours = 0.0
    elif refrigeration_hours is None:
        transit_hours = distance_km / max(average_speed_kmh, 1.0)
        refrigeration_hours = round(transit_hours + max(0.0, dwell_and_precool_hours), 2)

    # 5. Equation (1) from Du Plessis et al. (2023):
    #    Total fuel (L) = A * D_Total + B * (Weight_Cargo * D_Loaded) + C * T_Ref
    propulsion_fuel_l = (coeff_a * d_total) + (coeff_b * cargo_weight_tonnes * distance_km)
    reefer_fuel_l = coeff_c * refrigeration_hours
    total_fuel_l = propulsion_fuel_l + reefer_fuel_l

    # 6. Equation (2) from Du Plessis et al. (2023): Total emissions (kg CO2e) = Total fuel (L) * 3.24
    total_wtw_kg_co2e = total_fuel_l * DIESEL_WTW_KG_CO2E_PER_LITRE
    propulsion_wtw_kg = propulsion_fuel_l * DIESEL_WTW_KG_CO2E_PER_LITRE
    reefer_wtw_kg = reefer_fuel_l * DIESEL_WTW_KG_CO2E_PER_LITRE
    reefer_share_pct = (reefer_wtw_kg / total_wtw_kg_co2e * 100.0) if total_wtw_kg_co2e > 0 else 0.0

    # 7. Equation (3) & Equation (4) from Du Plessis et al. (2023)
    actual_tkm = cargo_weight_tonnes * distance_km
    emission_intensity_g_co2e_per_tkm = (total_wtw_kg_co2e * 1000.0) / actual_tkm
    product_carbon_footprint_kg_per_tonne = total_wtw_kg_co2e / cargo_weight_tonnes

    # Optional Du Plessis et al. (2024) Eq. (6) distribution loss uplift: CF_total = CF / (1 - Losses)
    loss_frac = max(0.0, distribution_loss_pct / 100.0 if distribution_loss_pct > 1.0 else distribution_loss_pct)
    if loss_frac >= 1.0:
        raise ValueError("distribution_loss_pct must be strictly less than 100%.")
    uplifted_cf_kg_per_tonne = product_carbon_footprint_kg_per_tonne / (1.0 - loss_frac)

    return {
        "methodology": {
            "primary_empirical_source": "Du Plessis, M.J., van Eeden, J., Goedhals-Gerber, L.L., & Else, J. (2023), Transp. Res. Part D 117, 103623",
            "accounting_standard": "BS EN 16258:2012 / GLEC Framework v3.1 (Well-to-Wheel)",
            "corridor_profile": coeffs["description"],
            "coefficients_applied": {
                "trailer_type": trailer_type,
                "route_direction": normalized_direction,
                "A_empty_truck_L_km": coeff_a,
                "B_payload_factor_L_tkm": coeff_b,
                "C_reefer_burn_L_hr": coeff_c,
                "wtw_emission_factor_kg_CO2e_L": DIESEL_WTW_KG_CO2E_PER_LITRE,
                "D_loaded_km": round(distance_km, 2),
                "D_empty_km": round(d_empty, 2),
                "D_total_km": round(d_total, 2),
                "empty_running_pct": round(empty_ratio * 100.0, 2)
            }
        },
        "fuel_breakdown_litres": {
            "propulsion_diesel_L": round(propulsion_fuel_l, 2),
            "refrigeration_tru_diesel_L": round(reefer_fuel_l, 2),
            "total_diesel_L": round(total_fuel_l, 2),
            "refrigeration_hours_used": round(refrigeration_hours, 2)
        },
        "emissions_kg_co2e": {
            "total_wtw_kg_co2e": round(total_wtw_kg_co2e, 2),
            "total_wtw_tonnes_co2e": round(total_wtw_kg_co2e / 1000.0, 4),
            "propulsion_wtw_kg_co2e": round(propulsion_wtw_kg, 2),
            "refrigeration_wtw_kg_co2e": round(reefer_wtw_kg, 2),
            "refrigeration_share_of_total_pct": round(reefer_share_pct, 1)
        },
        "glec_and_stellenbosch_kpis": {
            "transport_activity_tkm": round(actual_tkm, 1),
            "emission_intensity_g_co2e_per_tkm": round(emission_intensity_g_co2e_per_tkm, 2),
            "carbon_footprint_kg_CO2e_per_tonne_cargo": round(product_carbon_footprint_kg_per_tonne, 2),
            "carbon_footprint_with_loss_uplift_kg_CO2e_per_tonne": round(uplifted_cf_kg_per_tonne, 2),
            "empirical_accuracy_benchmark": "±6.7% fleet average error (±3.9% on calibrated long-haul routes)"
        }
    }
