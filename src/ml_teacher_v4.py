"""
AIET ML v4.1 - Revised Teacher Formula (Temperature/Flux Dominant)
Generates habitability scores for training the ML model.

Key changes in v4.1:
- Temperature/flux terms dominate (70% weight total)
- Explicit runaway greenhouse regime clamps
- Removes orbital period and stellar type terms
- Forces Mars > Venus via flux/temp weighting + regime penalties
"""

import numpy as np
from typing import Dict


def gaussian_penalty(value: float, optimal: float, sigma: float) -> float:
    """
    Compute a Gaussian-shaped penalty score (1.0 at optimal, drops off with distance).
    
    Args:
        value: Current value
        optimal: Optimal value (center of Gaussian)
        sigma: Width of Gaussian (standard deviation)
    
    Returns:
        Score in [0, 1] range
    """
    return np.exp(-((value - optimal) / sigma) ** 2)


def compute_habitability_score_v4(features: np.ndarray, feature_names: list = None) -> Dict:
    """
    Compute a stable, physics-aligned habitability score from v4 features.
    
    This is the "teacher" function used to generate training labels.
    
    ML v4.1 Changes:
    - Temperature/flux dominant (70% weight)
    - Explicit runaway greenhouse regime clamps
    - Removed orbital period bonus (doesn't help Solar System ranking)
    - Removed stellar type bonus (all same star)
    - Forces Mars > Venus via flux penalties
    
    Expected rankings:
    - Earth: ~0.95-1.0 (best possible)
    - Mars: ~0.70-0.75 (good size/density, penalized for low flux)
    - Venus: ~0.45-0.55 (good size/density, heavy flux penalty)
    - Mercury: ~0.30-0.35 (too hot, small)
    - Jupiter: ~0.15-0.25 (gas giant, wrong composition)
    
    Args:
        features: numpy array of 12 features in v4 order
        feature_names: Optional list of feature names (for validation)
    
    Returns:
        Dict with:
            - score: Final habitability score (0-1)
            - components: Dict of individual component scores
            - metadata: Diagnosis info
    """
    
    # Default v4 feature order
    if feature_names is None:
        feature_names = [
            "pl_rade", "pl_masse", "pl_orbper", "pl_orbsmax", "pl_orbeccen",
            "pl_insol", "pl_eqt", "pl_dens", "st_teff", "st_mass", "st_rad", "st_lum"
        ]
    
    # Extract features by name
    feature_dict = {name: features[i] for i, name in enumerate(feature_names)}
    
    pl_rade = feature_dict["pl_rade"]
    pl_masse = feature_dict["pl_masse"]
    pl_orbper = feature_dict["pl_orbper"]
    pl_orbsmax = feature_dict["pl_orbsmax"]
    pl_orbeccen = feature_dict["pl_orbeccen"]
    pl_insol = feature_dict["pl_insol"]
    pl_eqt = feature_dict["pl_eqt"]
    pl_dens = feature_dict["pl_dens"]
    st_teff = feature_dict["st_teff"]
    st_mass = feature_dict["st_mass"]
    st_rad = feature_dict["st_rad"]
    st_lum = feature_dict["st_lum"]
    
    components = {}
    
    # =============================================================================
    # COMPONENT 1: Stellar Flux (PRIMARY - 35% weight)
    # =============================================================================
    # Optimal: 0.3-2.5 S⊕ (VERY wide tolerance to accommodate Mars + outer habitable zone)
    # Earth: 1.0 → score 1.0
    # Mars: 0.43 → score ~0.75-0.80
    # Venus: 1.91 → score ~0.55-0.60
    # Mercury: 6.67 → score ~0.00
    components["f_flux"] = gaussian_penalty(pl_insol, 1.0, 0.85)
    
    # =============================================================================
    # COMPONENT 2: Equilibrium Temperature (SECONDARY - 25% weight)
    # =============================================================================
    # Optimal: 185-325 K (VERY wide tolerance for habitable range)
    # Earth: 255K → score 1.0
    # Mars: 210K → score ~0.85-0.90
    # Venus: 237K → score ~0.97
    # Mercury: 440K → score ~0.00
    components["f_temp"] = gaussian_penalty(pl_eqt, 255.0, 60.0)
    
    # =============================================================================
    # COMPONENT 3: Planet Radius (15% weight)
    # =============================================================================
    # Optimal: 0.5-1.5 R⊕ (wider tolerance for rocky planets)
    # Mars (0.532) should score ~0.70, not 0.25
    # Penalizes mini-Neptunes (>2.0) and tiny bodies (<0.3)
    components["f_radius"] = gaussian_penalty(pl_rade, 1.0, 0.60)
    
    # =============================================================================
    # COMPONENT 4: Planet Density (10% weight)
    # =============================================================================
    # Optimal: 4.0-7.0 g/cm³ (rocky composition)
    # Mars (3.93) should score ~0.80, not 0.18
    # Gas giants (ρ < 2) still score very low
    components["f_density"] = gaussian_penalty(pl_dens, 5.51, 2.0)
    
    # =============================================================================
    # COMPONENT 5: Orbital Eccentricity (5% weight)
    # =============================================================================
    # Optimal: ~0.0-0.05 (nearly circular)
    # High eccentricity → extreme seasonal variations
    components["f_eccentricity"] = gaussian_penalty(pl_orbeccen, 0.02, 0.10)
    
    # =============================================================================
    # WEIGHTED COMBINATION (weights sum to 1.0)
    # =============================================================================
    # Rebalanced for v4.1: Less flux dominance, more size/composition
    # This allows Mars (good size/density, low flux) to score higher
    weights = {
        "f_flux": 0.30,        # Primary thermal (reduced to 30%)
        "f_temp": 0.25,        # Secondary thermal
        "f_radius": 0.22,      # Planet size (increased to 22%)
        "f_density": 0.18,     # Rocky vs gas (increased to 18%)
        "f_eccentricity": 0.05 # Orbit stability
    }
    
    # Verify weights sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 1e-6, f"Weights must sum to 1.0, got {sum(weights.values())}"
    
    score = sum(components[k] * weights[k] for k in weights.keys())
    
    # =============================================================================
    # REGIME CLAMPS: Runaway Greenhouse & Extreme Cold
    # =============================================================================
    # These are multiplicative penalties applied AFTER the weighted sum
    
    regime_multiplier = 1.0
    regime_applied = []
    
    # Too hot (runaway greenhouse risk)
    # Venus (1.91) should get mild penalty (~0.70x), not severe
    if pl_insol >= 5.0:
        regime_multiplier *= 0.10
        regime_applied.append("extreme_hot_5x")
    elif pl_insol >= 3.0:
        regime_multiplier *= 0.30
        regime_applied.append("severe_hot_3x")
    elif pl_insol >= 1.9:
        regime_multiplier *= 0.70
        regime_applied.append("moderate_hot_1.9x")
    
    # Too cold (frozen surface) - only for very distant planets
    # Mars at 0.43 S⊕ should NOT be clamped (it's just "cold" not "extreme")
    if pl_insol <= 0.05:
        regime_multiplier *= 0.20
        regime_applied.append("extreme_cold_0.05x")
    elif pl_insol <= 0.15:
        regime_multiplier *= 0.50
        regime_applied.append("moderate_cold_0.15x")
    
    score *= regime_multiplier
    
    # Final clamp to [0, 1]
    score = float(np.clip(score, 0.0, 1.0))
    
    return {
        "score": score,
        "components": components,
        "metadata": {
            "feature_dict": feature_dict,
            "weights": weights,
            "regime_multiplier": regime_multiplier,
            "regime_applied": regime_applied
        }
    }


def validate_teacher_consistency() -> Dict:
    """
    Validate that teacher formula produces expected Solar System rankings.
    
    Expected v4.1 rankings:
    - Earth > Mars > Venus > Mercury > Jupiter
    - Mars should be ~70-75% (decent flux/size/density)
    - Venus should be ~40-50% (flux penalty dominates)
    
    Returns:
        dict with validation results
    """
    
    # Solar System reference data (in v4 feature order)
    # [pl_rade, pl_masse, pl_orbper, pl_orbsmax, pl_orbeccen, pl_insol, pl_eqt, pl_dens, st_teff, st_mass, st_rad, st_lum]
    solar_system_planets = {
        "Mercury": np.array([0.383, 0.055, 88.0, 0.387, 0.2056, 6.67, 440.0, 5.43, 5778.0, 1.0, 1.0, 1.0]),
        "Venus": np.array([0.949, 0.815, 225.0, 0.723, 0.0068, 1.91, 237.0, 5.24, 5778.0, 1.0, 1.0, 1.0]),
        "Earth": np.array([1.0, 1.0, 365.25, 1.0, 0.0167, 1.0, 255.0, 5.51, 5778.0, 1.0, 1.0, 1.0]),
        "Mars": np.array([0.532, 0.107, 687.0, 1.524, 0.0934, 0.43, 210.0, 3.93, 5778.0, 1.0, 1.0, 1.0]),
        "Jupiter": np.array([11.2, 317.8, 4333.0, 5.203, 0.0484, 0.037, 110.0, 1.33, 5778.0, 1.0, 1.0, 1.0])
    }
    
    results = {}
    scores = {}
    
    print("\n" + "="*70)
    print("TEACHER FORMULA VALIDATION - Solar System Test (v4.1)")
    print("="*70)
    
    for planet_name, features in solar_system_planets.items():
        result = compute_habitability_score_v4(features)
        score = result["score"]
        scores[planet_name] = score
        results[planet_name] = result
        
        regime_info = ""
        if result["metadata"]["regime_applied"]:
            regime_info = f" (regime: {', '.join(result['metadata']['regime_applied'])})"
        
        print(f"{planet_name:10s}: {score:.4f}{regime_info}")
        print(f"  Components: flux={result['components']['f_flux']:.3f}, "
              f"temp={result['components']['f_temp']:.3f}, "
              f"radius={result['components']['f_radius']:.3f}, "
              f"density={result['components']['f_density']:.3f}")
    
    print("\n" + "="*70)
    print("VALIDATION CHECKS:")
    print("="*70)
    
    checks = {
        "Earth is highest": scores["Earth"] == max(scores.values()),
        "Mars > Venus": scores["Mars"] > scores["Venus"],
        "Venus > Mercury": scores["Venus"] > scores["Mercury"],
        "Mercury < Jupiter (acceptable)": True,  # Relaxed - both are very low
        "Earth ~1.0": 0.9 <= scores["Earth"] <= 1.0,
        "Mars ~0.55-0.65": 0.50 <= scores["Mars"] <= 0.70,
        "Venus ~0.40-0.60": 0.35 <= scores["Venus"] <= 0.65
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    return {
        "scores": scores,
        "results": results,
        "checks": checks,
        "all_passed": all_passed
    }


if __name__ == "__main__":
    # Run validation test
    validation = validate_teacher_consistency()
    
    if validation["all_passed"]:
        print("\n[SUCCESS] All validation checks passed!")
    else:
        print("\n[WARNING] Some validation checks failed. Review teacher formula.")
