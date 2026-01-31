"""
AIET ML v4 - Solar System Validation Gates
Hard validation that model produces correct Solar System rankings
"""

import numpy as np
from typing import Dict, Callable, Any
import json
from datetime import datetime


def validate_solar_system_ranking(
    predict_fn: Callable,
    feature_builder_fn: Callable,
    export_path: str = None
) -> Dict:
    """
    Validate that model produces correct Solar System planet rankings.
    
    HARD GATES (must pass or training fails):
    1. Earth must be top-1 among Mercury/Venus/Earth/Mars
    2. Venus must score < 0.85 * Earth (penalize hot Venus)
    3. Jupiter must score < 0.5 * Earth (penalize gas giants)
    
    Args:
        predict_fn: Function that takes features array and returns score
        feature_builder_fn: Function that builds features from planet dict
        export_path: Optional path to export validation report JSON
    
    Returns:
        dict with validation results and pass/fail status
    """
    
    # Solar System reference data (NASA format)
    solar_system_data = {
        "Mercury": {
            "pl_rade": 0.383, "pl_masse": 0.055, "pl_orbper": 88.0, 
            "pl_orbsmax": 0.387, "pl_orbeccen": 0.2056, "pl_insol": 6.67,
            "pl_eqt": 440.0, "pl_dens": 5.43,
            "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
        },
        "Venus": {
            "pl_rade": 0.949, "pl_masse": 0.815, "pl_orbper": 225.0,
            "pl_orbsmax": 0.723, "pl_orbeccen": 0.0068, "pl_insol": 1.91,
            "pl_eqt": 327.0, "pl_dens": 5.24,
            "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
        },
        "Earth": {
            "pl_rade": 1.0, "pl_masse": 1.0, "pl_orbper": 365.25,
            "pl_orbsmax": 1.0, "pl_orbeccen": 0.0167, "pl_insol": 1.0,
            "pl_eqt": 255.0, "pl_dens": 5.51,
            "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
        },
        "Mars": {
            "pl_rade": 0.532, "pl_masse": 0.107, "pl_orbper": 687.0,
            "pl_orbsmax": 1.524, "pl_orbeccen": 0.0934, "pl_insol": 0.43,
            "pl_eqt": 210.0, "pl_dens": 3.93,
            "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
        },
        "Jupiter": {
            "pl_rade": 11.2, "pl_masse": 317.8, "pl_orbper": 4333.0,
            "pl_orbsmax": 5.203, "pl_orbeccen": 0.0484, "pl_insol": 0.037,
            "pl_eqt": 110.0, "pl_dens": 1.33,
            "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
        }
    }
    
    # Build features and predict scores
    scores = {}
    feature_vectors = {}
    
    print("\n" + "="*70)
    print("SOLAR SYSTEM VALIDATION")
    print("="*70)
    print("\nPredicting scores...")
    
    for planet_name, planet_data in solar_system_data.items():
        # Build features
        features, meta = feature_builder_fn(planet_data, return_meta=True)
        feature_vectors[planet_name] = features
        
        # Predict score
        score = predict_fn(features)
        scores[planet_name] = float(score)
        
        print(f"  {planet_name:10s}: {score:.4f}")
    
    # =============================================================================
    # VALIDATION GATES
    # =============================================================================
    
    gates = {}
    
    # Gate 1: Earth must be top among rocky inner planets
    rocky_planets = ["Mercury", "Venus", "Earth", "Mars"]
    rocky_scores = {k: scores[k] for k in rocky_planets}
    rocky_ranking = sorted(rocky_scores.items(), key=lambda x: x[1], reverse=True)
    top_rocky_planet = rocky_ranking[0][0]
    
    gates["earth_top_rocky"] = {
        "pass": (top_rocky_planet == "Earth"),
        "description": "Earth must be top-1 among rocky inner planets",
        "ranking": rocky_ranking,
        "top_planet": top_rocky_planet
    }
    
    # Gate 2: Mars > Venus (NEW v4.1 requirement)
    mars_greater = scores["Mars"] > scores["Venus"]
    gates["mars_gt_venus"] = {
        "pass": mars_greater,
        "description": "Mars must score > Venus (thermal dominance)",
        "mars_score": scores["Mars"],
        "venus_score": scores["Venus"],
        "difference": scores["Mars"] - scores["Venus"]
    }
    
    # Gate 3: Venus must score significantly lower than Earth (TIGHTENED v4.1)
    venus_ratio = scores["Venus"] / scores["Earth"] if scores["Earth"] > 0 else 999
    gates["venus_penalty"] = {
        "pass": (venus_ratio < 0.55),
        "description": "Venus must score < 0.55 * Earth (hot planet penalty)",
        "venus_score": scores["Venus"],
        "earth_score": scores["Earth"],
        "ratio": venus_ratio,
        "threshold": 0.55
    }
    
    # Gate 4: Mercury penalty (NEW v4.1)
    mercury_ratio = scores["Mercury"] / scores["Earth"] if scores["Earth"] > 0 else 999
    gates["mercury_penalty"] = {
        "pass": (mercury_ratio < 0.35),
        "description": "Mercury must score < 0.35 * Earth (extreme heat penalty)",
        "mercury_score": scores["Mercury"],
        "earth_score": scores["Earth"],
        "ratio": mercury_ratio,
        "threshold": 0.35
    }
    
    # Gate 5: Jupiter must score low (Earth-centric habitability)
    jupiter_ratio = scores["Jupiter"] / scores["Earth"] if scores["Earth"] > 0 else 999
    gates["jupiter_penalty"] = {
        "pass": (jupiter_ratio < 0.5),
        "description": "Jupiter must score < 0.5 * Earth (gas giant penalty)",
        "jupiter_score": scores["Jupiter"],
        "earth_score": scores["Earth"],
        "ratio": jupiter_ratio,
        "threshold": 0.5
    }
    
    # Overall pass/fail
    all_pass = all(gate["pass"] for gate in gates.values())
    
    # =============================================================================
    # PRINT RESULTS
    # =============================================================================
    
    print("\n" + "="*70)
    print("VALIDATION GATES")
    print("="*70)
    
    for gate_name, gate_data in gates.items():
        status = "[PASS]" if gate_data["pass"] else "[FAIL]"
        print(f"\n{status} {gate_data['description']}")
        
        if gate_name == "earth_top_rocky":
            print(f"  Top planet: {gate_data['top_planet']}")
            print(f"  Ranking:")
            for planet, score in gate_data['ranking']:
                print(f"    {planet:10s}: {score:.4f}")
        
        elif gate_name == "mars_gt_venus":
            print(f"  Mars: {gate_data['mars_score']:.4f}")
            print(f"  Venus: {gate_data['venus_score']:.4f}")
            print(f"  Difference: {gate_data['difference']:.4f}")
        
        elif gate_name in ["venus_penalty", "mercury_penalty", "jupiter_penalty"]:
            planet_name = gate_name.split('_')[0].title()
            planet_score_key = f"{gate_name.split('_')[0]}_score"
            print(f"  Ratio: {gate_data['ratio']:.4f} (threshold: {gate_data['threshold']:.2f})")
            print(f"  Earth: {gate_data['earth_score']:.4f}")
            print(f"  {planet_name}: {gate_data[planet_score_key]:.4f}")
    
    print("\n" + "="*70)
    if all_pass:
        print("[PASS] ALL VALIDATION GATES PASSED")
    else:
        print("[FAIL] VALIDATION FAILED - Model does not produce correct Solar System rankings")
    print("="*70)
    
    # =============================================================================
    # EXPORT REPORT
    # =============================================================================
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "validation_version": "v4",
        "scores": scores,
        "gates": gates,
        "all_pass": all_pass
    }
    
    if export_path:
        with open(export_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nValidation report exported to: {export_path}")
    
    return report


def validate_model_predictions(
    model: Any,
    feature_builder_fn: Callable,
    export_dir: str = "exports"
) -> bool:
    """
    Convenience wrapper for validating trained model.
    
    Args:
        model: Trained model with predict() method
        feature_builder_fn: Function that builds features from planet dict
        export_dir: Directory to export validation report
    
    Returns:
        bool: True if all gates pass, False otherwise
    """
    
    # Create predict function
    def predict_fn(features):
        # Handle both XGBRegressor and Booster
        try:
            features_2d = features.reshape(1, -1)
            pred = model.predict(features_2d)
            # Handle both scalar and array outputs
            return pred[0] if hasattr(pred, '__getitem__') else pred
        except Exception as e:
            raise ValueError(f"Model prediction failed: {e}")
    
    # Generate export path
    import os
    os.makedirs(export_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(export_dir, f"validation_v4_{timestamp}.json")
    
    # Run validation
    report = validate_solar_system_ranking(
        predict_fn=predict_fn,
        feature_builder_fn=feature_builder_fn,
        export_path=export_path
    )
    
    return report["all_pass"]


if __name__ == "__main__":
    # Test with teacher formula
    print("Testing validation gates with teacher formula...")
    
    from ml_teacher_v4 import compute_habitability_score_v4
    from ml_features_v4 import build_features_v4
    
    def teacher_predict(features):
        result = compute_habitability_score_v4(features)
        return result["score"]
    
    report = validate_solar_system_ranking(
        predict_fn=teacher_predict,
        feature_builder_fn=build_features_v4
    )
    
    if report["all_pass"]:
        print("\n[SUCCESS] Teacher formula passes all validation gates")
        exit(0)
    else:
        print("\n[ERROR] Teacher formula fails validation gates")
        exit(1)
