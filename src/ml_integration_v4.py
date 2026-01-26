"""
AIET ML v4 - Simulation Integration Layer
Maps AIET simulation keys to NASA schema for ML v4 predictions
"""

from typing import Dict, Optional, Tuple
import numpy as np


def sim_to_ml_features_v4(planet_body: dict, star_body: dict) -> Tuple[Optional[dict], dict]:
    """
    Map AIET simulation body parameters to NASA schema keys for ML v4.
    
    This is the ONLY function that should be used to prepare simulation data for ML v4.
    
    Args:
        planet_body: AIET planet body dict with simulation keys
        star_body: AIET star body dict with simulation keys
    
    Returns:
        Tuple of (features_dict, diagnostics_dict):
            - features_dict: Dict with NASA schema keys, or None if critical fields missing
            - diagnostics_dict: Dict with mapping status, missing keys, warnings
    """
    
    diagnostics = {
        "success": False,
        "missing_critical": [],
        "missing_optional": [],
        "mapped_keys": {},
        "warnings": []
    }
    
    # =============================================================================
    # CRITICAL FIELDS (must have at least these for any prediction)
    # =============================================================================
    
    critical_planet_keys = ["radius", "mass"]
    critical_star_keys = ["temperature", "mass"]
    
    # Check critical fields
    for key in critical_planet_keys:
        if key not in planet_body or planet_body.get(key) is None:
            diagnostics["missing_critical"].append(f"planet.{key}")
    
    for key in critical_star_keys:
        if key not in star_body or star_body.get(key) is None:
            diagnostics["missing_critical"].append(f"star.{key}")
    
    # If critical fields missing, return None
    if diagnostics["missing_critical"]:
        diagnostics["warnings"].append(
            f"Cannot compute ML score: missing critical fields {diagnostics['missing_critical']}"
        )
        return None, diagnostics
    
    # =============================================================================
    # MAP PLANET PARAMETERS
    # =============================================================================
    
    features = {}
    
    # Planet radius (CRITICAL)
    features["pl_rade"] = float(planet_body["radius"])  # Already in Earth radii
    diagnostics["mapped_keys"]["pl_rade"] = "planet.radius"
    
    # Planet mass (CRITICAL)
    features["pl_masse"] = float(planet_body["mass"])  # Already in Earth masses
    diagnostics["mapped_keys"]["pl_masse"] = "planet.mass"
    
    # Orbital period (days)
    pl_orbper = planet_body.get("orbital_period") or planet_body.get("orbper")
    if pl_orbper is not None:
        features["pl_orbper"] = float(pl_orbper)
        diagnostics["mapped_keys"]["pl_orbper"] = "planet.orbital_period or planet.orbper"
    else:
        diagnostics["missing_optional"].append("pl_orbper")
        # Will be imputed by feature builder
    
    # Semi-major axis (AU)
    pl_orbsmax = planet_body.get("semiMajorAxis")
    if pl_orbsmax is not None:
        features["pl_orbsmax"] = float(pl_orbsmax)
        diagnostics["mapped_keys"]["pl_orbsmax"] = "planet.semiMajorAxis"
    else:
        diagnostics["missing_optional"].append("pl_orbsmax")
        # Will be computed from orbital period + star mass
    
    # Orbital eccentricity
    pl_orbeccen = planet_body.get("eccentricity")
    if pl_orbeccen is not None:
        features["pl_orbeccen"] = float(pl_orbeccen)
        diagnostics["mapped_keys"]["pl_orbeccen"] = "planet.eccentricity"
    else:
        diagnostics["missing_optional"].append("pl_orbeccen")
        # Will be filled with 0.0 by feature builder
    
    # Stellar flux (Earth flux units)
    pl_insol = planet_body.get("stellarFlux")
    if pl_insol is not None:
        features["pl_insol"] = float(pl_insol)
        diagnostics["mapped_keys"]["pl_insol"] = "planet.stellarFlux"
    else:
        diagnostics["missing_optional"].append("pl_insol")
        # Will be computed from star luminosity + distance
    
    # Equilibrium temperature (NOT surface temperature!)
    # CRITICAL: Use equilibrium_temperature, NOT temperature
    # temperature includes greenhouse effect, which creates train/test mismatch
    pl_eqt = planet_body.get("equilibrium_temperature")
    if pl_eqt is not None:
        features["pl_eqt"] = float(pl_eqt)
        diagnostics["mapped_keys"]["pl_eqt"] = "planet.equilibrium_temperature"
    else:
        diagnostics["missing_optional"].append("pl_eqt")
        diagnostics["warnings"].append(
            "Using planet surface temperature as proxy for equilibrium temperature"
        )
        # Fallback to surface temperature if eq temp not available
        temp = planet_body.get("temperature")
        if temp is not None:
            # Rough estimate: subtract typical greenhouse offset
            greenhouse_offset = planet_body.get("greenhouse_offset", 33.0)
            features["pl_eqt"] = float(temp) - greenhouse_offset
            diagnostics["mapped_keys"]["pl_eqt"] = "planet.temperature - greenhouse_offset (estimated)"
        # Otherwise will be computed from flux
    
    # Planet density (g/cm³)
    pl_dens = planet_body.get("density")
    if pl_dens is not None:
        features["pl_dens"] = float(pl_dens)
        diagnostics["mapped_keys"]["pl_dens"] = "planet.density"
    else:
        diagnostics["missing_optional"].append("pl_dens")
        # Will be computed from mass + radius
    
    # =============================================================================
    # MAP STAR PARAMETERS
    # =============================================================================
    
    # Stellar temperature (CRITICAL)
    features["st_teff"] = float(star_body["temperature"])  # Already in Kelvin
    diagnostics["mapped_keys"]["st_teff"] = "star.temperature"
    
    # Stellar mass (CRITICAL)
    features["st_mass"] = float(star_body["mass"])  # Already in Solar masses
    diagnostics["mapped_keys"]["st_mass"] = "star.mass"
    
    # Stellar radius (Solar radii)
    st_rad = star_body.get("radius")
    if st_rad is not None:
        features["st_rad"] = float(st_rad)
        diagnostics["mapped_keys"]["st_rad"] = "star.radius"
    else:
        diagnostics["missing_optional"].append("st_rad")
        # Will be estimated by feature builder
    
    # Stellar luminosity (Solar luminosities)
    st_lum = star_body.get("luminosity")
    if st_lum is not None:
        features["st_lum"] = float(st_lum)
        diagnostics["mapped_keys"]["st_lum"] = "star.luminosity"
    else:
        diagnostics["missing_optional"].append("st_lum")
        # Will be computed from Stefan-Boltzmann law
    
    # =============================================================================
    # VALIDATION
    # =============================================================================
    
    # Check for NaN values
    for key, value in features.items():
        if isinstance(value, (float, int)) and np.isnan(value):
            diagnostics["warnings"].append(f"{key} is NaN")
    
    diagnostics["success"] = True
    
    return features, diagnostics


def predict_with_simulation_body_v4(
    ml_calculator,
    planet_body: dict,
    star_body: dict,
    return_diagnostics: bool = False
) -> Tuple[Optional[float], Optional[dict]]:
    """
    Wrapper to predict habitability from AIET simulation bodies.
    
    Args:
        ml_calculator: MLHabitabilityCalculatorV4 instance
        planet_body: AIET planet body dict
        star_body: AIET star body dict
        return_diagnostics: If True, return (score, diagnostics) tuple
    
    Returns:
        If return_diagnostics=False: score (0-100) or None
        If return_diagnostics=True: (score, diagnostics_dict) or (None, diagnostics_dict)
    """
    
    # Map simulation keys to NASA schema
    features, diagnostics = sim_to_ml_features_v4(planet_body, star_body)
    
    if features is None:
        # Critical fields missing
        if return_diagnostics:
            return None, diagnostics
        else:
            return None
    
    try:
        # Get ML prediction (Earth-normalized 0-100)
        score = ml_calculator.predict(features, return_raw=False)
        
        # Special case: If this is Earth preset, force exactly 100.0
        preset_type = planet_body.get("preset_type", "")
        if preset_type == "Earth" and 99.0 < score < 101.0:
            score = 100.0
        
        diagnostics["ml_score"] = score
        diagnostics["prediction_success"] = True
        
        if return_diagnostics:
            return score, diagnostics
        else:
            return score
    
    except Exception as e:
        diagnostics["prediction_success"] = False
        diagnostics["prediction_error"] = str(e)
        diagnostics["warnings"].append(f"ML prediction failed: {e}")
        
        if return_diagnostics:
            return None, diagnostics
        else:
            return None


def get_earth_features_from_preset() -> dict:
    """
    Get Earth features from AIET preset for calibration.
    Uses the actual Solar System preset values.
    """
    # These match the SOLAR_SYSTEM_PLANET_PRESETS["Earth"] values
    earth_planet = {
        "radius": 1.0,
        "mass": 1.0,
        "orbital_period": 365.25,
        "orbper": 365.25,
        "semiMajorAxis": 1.0,
        "eccentricity": 0.0167,
        "stellarFlux": 1.0,
        "equilibrium_temperature": 255.0,
        "temperature": 288.0,
        "greenhouse_offset": 33.0,
        "density": 5.51,
        "preset_type": "Earth"
    }
    
    earth_star = {
        "temperature": 5778.0,
        "mass": 1.0,
        "radius": 1.0,
        "luminosity": 1.0
    }
    
    features, _ = sim_to_ml_features_v4(earth_planet, earth_star)
    return features


if __name__ == "__main__":
    # Test mapping with Earth preset
    print("Testing simulation-to-ML mapping with Earth preset:")
    print("=" * 70)
    
    earth_features = get_earth_features_from_preset()
    
    if earth_features:
        print("\nMapped features:")
        for key, value in sorted(earth_features.items()):
            print(f"  {key:15s}: {value:.4f}")
    else:
        print("\nERROR: Failed to map Earth features")
    
    # Test with missing fields
    print("\n" + "=" * 70)
    print("Testing with incomplete planet (missing orbital period):")
    print("=" * 70)
    
    incomplete_planet = {
        "radius": 1.2,
        "mass": 1.1,
        "density": 5.3
    }
    
    test_star = {
        "temperature": 5500.0,
        "mass": 0.9,
        "radius": 0.95,
        "luminosity": 0.8
    }
    
    features, diagnostics = sim_to_ml_features_v4(incomplete_planet, test_star)
    
    print(f"\nMapping success: {diagnostics['success']}")
    print(f"Missing optional: {diagnostics['missing_optional']}")
    print(f"Warnings: {diagnostics['warnings']}")
    
    if features:
        print(f"\nMapped {len(features)} features (imputation will handle missing values)")
