"""
AIET ML v4 - Simulation Integration Layer
Maps AIET simulation keys to NASA schema for ML v4 predictions

SINGLE SOURCE OF TRUTH for feature extraction from simulation bodies.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import json
import os
from datetime import datetime
from surface_classification import classify_surface, get_display_label, should_display_score


# =============================================================================
# HARD VALIDATION RANGES (Unit/Range Guardrails)
# =============================================================================

FEATURE_VALIDATION_RANGES = {
    "pl_rade": (0.05, 25.0, "R⊕"),  # Earth radii
    "pl_masse": (1e-4, 1e4, "M⊕"),  # Earth masses
    "pl_orbper": (0.05, 1e7, "days"),  # Orbital period
    "pl_orbsmax": (0.01, 1e5, "AU"),  # Semi-major axis
    "pl_orbeccen": (0.0, 0.999, ""),  # Eccentricity (unitless)
    "pl_insol": (1e-5, 1e3, "S⊕"),  # Stellar flux
    "pl_eqt": (10.0, 2000.0, "K"),  # Equilibrium temperature
    "pl_dens": (0.1, 20.0, "g/cm³"),  # Density
    "st_teff": (2000.0, 50000.0, "K"),  # Stellar temperature
    "st_mass": (0.08, 100.0, "M☉"),  # Stellar mass
    "st_rad": (0.1, 1000.0, "R☉"),  # Stellar radius
    "st_lum": (1e-5, 1e6, "L☉"),  # Stellar luminosity
}


def validate_feature_value(key: str, value: float) -> Tuple[bool, Optional[str]]:
    """
    Validate that a feature value is within expected physical ranges.
    
    Args:
        key: Feature name (e.g., "pl_rade")
        value: Feature value
    
    Returns:
        (is_valid, error_message)
    """
    if key not in FEATURE_VALIDATION_RANGES:
        return True, None  # Unknown key - allow through
    
    min_val, max_val, unit = FEATURE_VALIDATION_RANGES[key]
    
    if not np.isfinite(value):
        return False, f"{key} is not finite (got {value})"
    
    if value < min_val or value > max_val:
        unit_str = f" {unit}" if unit else ""
        return False, f"{key} = {value:.4e}{unit_str} outside valid range [{min_val:.2e}, {max_val:.2e}]{unit_str}"
    
    return True, None


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
    # ASSERTION: Must be from body["radius"] (Earth radii), NOT actual_radius
    pl_rade = float(planet_body["radius"])  # Already in Earth radii
    
    # Debug assertion: Check for unit mismatches
    if pl_rade > 25.0:
        diagnostics["warnings"].append(
            f"UNIT MISMATCH WARNING: pl_rade={pl_rade:.2f} > 25 R_E. "
            "Check if using actual_radius (km/m) instead of radius (R_E)."
        )
    
    features["pl_rade"] = pl_rade
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
    # ASSERTION: Must be in g/cm³, NOT kg/m³
    pl_dens = planet_body.get("density")
    if pl_dens is not None:
        pl_dens = float(pl_dens)
        
        # Debug assertion: Check for unit mismatches
        if pl_dens > 20.0:
            diagnostics["warnings"].append(
                f"UNIT MISMATCH WARNING: pl_dens={pl_dens:.2f} > 20 g/cm^3. "
                "Check if using kg/m^3 (divide by 1000 to get g/cm^3)."
            )
        
        features["pl_dens"] = pl_dens
        diagnostics["mapped_keys"]["pl_dens"] = "planet.density"
    else:
        pl_dens = None  # Will be computed from mass + radius
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


def planet_star_to_features_v4_canonical(
    planet_body: dict,
    star_body: dict
) -> Tuple[Optional[dict], dict]:
    """
    CANONICAL ADAPTER: Single source of truth for sim → ML feature extraction.
    
    This function enforces:
    1. Correct unit extraction (R⊕ not km, g/cm³ not kg/m³)
    2. Hard validation of all features
    3. Explicit tracking of direct vs computed vs imputed
    4. Returns None + error on validation failure (never returns invalid features)
    
    Args:
        planet_body: AIET planet body dict
        star_body: AIET star body dict
    
    Returns:
        (features_dict, meta_dict):
            - features_dict: NASA schema keys with validated values, or None if validation fails
            - meta_dict: {
                "success": bool,
                "validation_errors": list of validation error messages,
                "mapping_errors": list of mapping error messages,
                "feature_sources": dict mapping feature -> "direct" | "computed" | "imputed",
                "warnings": list of non-critical warnings
            }
    """
    meta = {
        "success": False,
        "validation_errors": [],
        "mapping_errors": [],
        "feature_sources": {},
        "warnings": []
    }
    
    # Step 1: Map simulation keys to NASA schema (using existing adapter)
    features, diagnostics = sim_to_ml_features_v4(planet_body, star_body)
    
    if features is None:
        # Critical fields missing
        meta["mapping_errors"] = diagnostics.get("warnings", [])
        meta["mapping_errors"].extend([f"Missing: {k}" for k in diagnostics.get("missing_critical", [])])
        return None, meta
    
    # Step 2: Hard validation of all feature values
    for key, value in features.items():
        is_valid, error_msg = validate_feature_value(key, value)
        if not is_valid:
            meta["validation_errors"].append(error_msg)
    
    # If any validation errors, fail hard (do NOT return invalid features)
    if meta["validation_errors"]:
        return None, meta
    
    # Step 3: Track feature sources (direct from sim, computed, or imputed)
    # Populate from diagnostics
    for key in features.keys():
        # Check if this key was mapped directly
        if key in diagnostics.get("mapped_keys", {}):
            source_path = diagnostics["mapped_keys"][key]
            if "Kepler" in source_path or "from" in source_path or "Stefan-Boltzmann" in source_path:
                meta["feature_sources"][key] = "computed"
            else:
                meta["feature_sources"][key] = "direct"
        else:
            # Will be imputed by feature builder
            meta["feature_sources"][key] = "imputed"
    
    # Step 4: Copy over warnings from diagnostics
    meta["warnings"] = diagnostics.get("warnings", [])
    
    meta["success"] = True
    return features, meta


def predict_with_simulation_body_v4(
    ml_calculator,
    planet_body: dict,
    star_body: dict,
    return_diagnostics: bool = False,
    surface_mode: str = "all"
) -> Tuple[Optional[float], Optional[dict]]:
    """
    Wrapper to predict habitability from AIET simulation bodies.
    
    Args:
        ml_calculator: MLHabitabilityCalculatorV4 instance
        planet_body: AIET planet body dict
        star_body: AIET star body dict
        return_diagnostics: If True, return (score, diagnostics) tuple
        surface_mode: "all" (show scores for all) or "rocky_only" (giants show None)
    
    Returns:
        If return_diagnostics=False: score (0-100) or None
        If return_diagnostics=True: (score, diagnostics_dict) or (None, diagnostics_dict)
    
    Meta fields added to diagnostics:
        - surface_class: "rocky" | "giant" | "unknown"
        - surface_applicable: bool
        - surface_reason: str
        - surface_warnings: list[str]
        - display_label: str (UI badge text)
        - should_display_score: bool (True = show percent, False = show "—")
        - score_raw: float (raw ML score before normalization)
        - score_display: float or None (Earth-normalized 0-100, or None if shouldn't display)
    """
    
    # Map simulation keys to NASA schema
    features, diagnostics = sim_to_ml_features_v4(planet_body, star_body)
    
    if features is None:
        # Critical fields missing - return None with diagnostics
        diagnostics["score_raw"] = None
        diagnostics["score_display"] = None
        diagnostics["surface_class"] = "unknown"
        diagnostics["surface_applicable"] = False
        diagnostics["surface_reason"] = "Missing critical fields"
        diagnostics["surface_warnings"] = diagnostics.get("warnings", [])
        diagnostics["display_label"] = "Data Incomplete"
        diagnostics["should_display_score"] = False
        
        if return_diagnostics:
            return None, diagnostics
        else:
            return None
    
    # =============================================================================
    # SURFACE CLASSIFICATION (pure function, no side effects)
    # =============================================================================
    
    # Get radius and density for classification
    pl_rade = features.get("pl_rade")
    pl_dens = features.get("pl_dens")
    
    # Classify surface type
    surface_info = classify_surface(pl_rade, pl_dens)
    
    # Add surface classification to diagnostics
    diagnostics["surface_class"] = surface_info["surface_class"]
    diagnostics["surface_applicable"] = surface_info["surface_applicable"]
    diagnostics["surface_reason"] = surface_info["reason"]
    diagnostics["surface_warnings"] = surface_info["warnings"]
    diagnostics["display_label"] = get_display_label(
        surface_info["surface_class"], 
        surface_mode
    )
    diagnostics["should_display_score"] = should_display_score(
        surface_info["surface_class"], 
        surface_mode
    )
    
    # =============================================================================
    # ML PREDICTION (never returns 0.0 on exception)
    # =============================================================================
    
    try:
        # Get ML prediction (raw and Earth-normalized)
        score_raw = ml_calculator.predict(features, return_raw=True)
        score_normalized = ml_calculator.predict(features, return_raw=False)
        
        # Special case: If this is Earth preset, force exactly 100.0
        preset_type = planet_body.get("preset_type", "")
        if preset_type == "Earth" and 99.0 < score_normalized < 101.0:
            score_normalized = 100.0
        
        # Store both raw and normalized scores
        diagnostics["score_raw"] = score_raw
        diagnostics["prediction_success"] = True
        
        # =============================================================================
        # DISPLAY POLICY (based on surface_mode)
        # =============================================================================
        
        if diagnostics["should_display_score"]:
            # Show numeric score (rocky planet or surface_mode="all")
            diagnostics["score_display"] = score_normalized
            final_score = score_normalized
        else:
            # Hide numeric score (giant in surface_mode="rocky_only")
            diagnostics["score_display"] = None
            final_score = None
        
        if return_diagnostics:
            return final_score, diagnostics
        else:
            return final_score
    
    except Exception as e:
        # NEVER return 0.0 on exception - return None with error info
        diagnostics["prediction_success"] = False
        diagnostics["prediction_error"] = str(e)
        diagnostics["warnings"].append(f"ML prediction failed: {e}")
        diagnostics["score_raw"] = None
        diagnostics["score_display"] = None
        
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


def export_ml_debug_snapshot(
    ml_calculator,
    bodies_list: list,
    output_path: str = None
):
    """
    Export debug snapshot of ML scoring for all planets.
    
    Args:
        ml_calculator: MLHabitabilityCalculatorV4 instance
        bodies_list: List of (planet_body, star_body, name) tuples
        output_path: Path to save JSON file (default: exports/ml_debug_snapshot.json)
    
    Creates a JSON file with:
        - name, pl_rade, pl_dens
        - surface_class, surface_warnings
        - score_raw, score_display
        - all diagnostic info
    """
    import json
    import os
    from datetime import datetime
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"exports/ml_debug_snapshot_{timestamp}.json"
    
    # Ensure exports directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "planets": []
    }
    
    for planet_body, star_body, name in bodies_list:
        score, diagnostics = predict_with_simulation_body_v4(
            ml_calculator,
            planet_body,
            star_body,
            return_diagnostics=True,
            surface_mode="all"  # Always compute scores for debug
        )
        
        planet_data = {
            "name": name,
            "pl_rade": planet_body.get("radius"),
            "pl_dens": planet_body.get("density"),
            "surface_class": diagnostics.get("surface_class", "unknown"),
            "surface_applicable": diagnostics.get("surface_applicable", False),
            "surface_reason": diagnostics.get("surface_reason", ""),
            "surface_warnings": diagnostics.get("surface_warnings", []),
            "score_raw": diagnostics.get("score_raw"),
            "score_display": diagnostics.get("score_display"),
            "display_label": diagnostics.get("display_label", ""),
            "input_warnings": diagnostics.get("warnings", []),
            "missing_critical": diagnostics.get("missing_critical", []),
            "missing_optional": diagnostics.get("missing_optional", []),
            "prediction_success": diagnostics.get("prediction_success", False),
            "prediction_error": diagnostics.get("prediction_error", None)
        }
        
        snapshot["planets"].append(planet_data)
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"\n[DEBUG] ML snapshot exported to: {output_path}")
    return output_path


def export_ml_snapshot_single_planet(
    ml_calculator,
    planet_body: dict,
    star_body: dict,
    output_path: str = None
) -> str:
    """
    Export auditable ML snapshot for a single planet.
    
    CRITICAL FOR DEMO: Shows exactly what features were fed to model and what score came out.
    
    Args:
        ml_calculator: MLHabitabilityCalculatorV4 instance
        planet_body: AIET planet body dict
        star_body: AIET star body dict
        output_path: Optional output path (default: exports/ml_snapshot_<timestamp>.json)
    
    Returns:
        Path to exported JSON file
    
    Exports:
        - Model artifact path
        - Earth reference raw score
        - All 12 features actually fed to model
        - Feature sources (direct/computed/imputed)
        - Validation status
        - Raw score (0-1)
        - Displayed score (0-100)
        - Planet name + ID
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        planet_name = planet_body.get("name", "planet").replace(" ", "_")
        output_path = f"exports/ml_snapshot_{planet_name}_{timestamp}.json"
    
    # Ensure exports directory exists
    os.makedirs("exports", exist_ok=True)
    
    # Use canonical adapter for feature extraction
    features, meta = planet_star_to_features_v4_canonical(planet_body, star_body)
    
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "planet": {
            "name": planet_body.get("name", "Unknown"),
            "id": planet_body.get("id", "N/A")
        },
        "model": {
            "version": "v4.1",
            "type": "XGBoost",
            "earth_reference_raw": getattr(ml_calculator, 'earth_raw_score', None)
        },
        "feature_extraction": {
            "success": meta["success"],
            "validation_errors": meta.get("validation_errors", []),
            "mapping_errors": meta.get("mapping_errors", []),
            "warnings": meta.get("warnings", [])
        },
        "features": {},
        "feature_sources": meta.get("feature_sources", {}),
        "prediction": {
            "raw_score": None,
            "displayed_score": None,
            "success": False,
            "error": None
        }
    }
    
    # Add features if extraction succeeded
    if features:
        snapshot["features"] = {k: float(v) for k, v in features.items()}
        
        # Try prediction
        try:
            from ml_features_v4 import build_features_v4
            feature_vector, feature_meta = build_features_v4(features, return_meta=True)
            
            # Get raw score
            raw_score = ml_calculator._predict_raw(feature_vector)
            
            # Get normalized score
            normalized_score = (raw_score / ml_calculator.earth_raw_score) * 100.0 if ml_calculator.earth_raw_score > 0 else raw_score * 100.0
            normalized_score = float(np.clip(normalized_score, 0.0, 100.0))
            
            snapshot["prediction"]["raw_score"] = float(raw_score)
            snapshot["prediction"]["displayed_score"] = float(normalized_score)
            snapshot["prediction"]["success"] = True
            snapshot["imputed_by_feature_builder"] = feature_meta.get("imputed_fields", [])
            
        except Exception as e:
            snapshot["prediction"]["error"] = str(e)
            snapshot["prediction"]["success"] = False
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"\n[ML SNAPSHOT] Exported to: {output_path}")
    if snapshot["prediction"]["success"]:
        print(f"  Planet: {snapshot['planet']['name']}")
        print(f"  Raw Score: {snapshot['prediction']['raw_score']:.6f}")
        print(f"  Displayed Score: {snapshot['prediction']['displayed_score']:.2f}%")
    else:
        print(f"  Status: FAILED")
        if snapshot["feature_extraction"]["validation_errors"]:
            print(f"  Validation Errors: {snapshot['feature_extraction']['validation_errors']}")
        if snapshot["feature_extraction"]["mapping_errors"]:
            print(f"  Mapping Errors: {snapshot['feature_extraction']['mapping_errors']}")
    
    return output_path


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
    
    # Test surface classification
    print("\n" + "=" * 70)
    print("Testing surface classification:")
    print("=" * 70)
    
    from surface_classification import classify_surface
    
    test_cases = [
        ("Earth", 1.0, 5.51),
        ("Mars", 0.532, 3.93),
        ("Jupiter", 11.2, 1.33),
        ("Neptune", 3.9, 1.64),
        ("Super-Earth", 1.5, 4.0),
        ("Mini-Neptune", 2.5, 2.0),
    ]
    
    for name, rade, dens in test_cases:
        result = classify_surface(rade, dens)
        print(f"\n{name:15s}: {result['surface_class']:8s} (applicable: {result['surface_applicable']})")
        print(f"  Reason: {result['reason']}")
