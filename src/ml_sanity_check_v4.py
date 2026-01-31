"""
AIET ML v4 - Sanity Check Tool
Validates ML v4 predictions against expected behavior
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any

from ml_features_v4 import build_features_v4, load_feature_schema, validate_features
from ml_validation_v4 import validate_solar_system_ranking


def run_ml_sanity_check_v4(
    ml_calculator: Any,
    planet_features: dict = None,
    export_dir: str = "exports"
) -> Dict:
    """
    Run comprehensive ML v4 sanity checks.
    
    Args:
        ml_calculator: MLHabitabilityCalculatorV4 instance
        planet_features: Optional dict with planet parameters to test
                        If None, uses Earth as reference
        export_dir: Directory to export JSON report
    
    Returns:
        dict: Comprehensive diagnostic report with PASS/WARN/FAIL status
    """
    
    # Initialize report
    report = {
        "timestamp": datetime.now().isoformat(),
        "ml_version": "v4",
        "checks": {},
        "overall_status": "PASS",
        "recommended_fix": "No issues detected"
    }
    
    # Load feature schema
    try:
        feature_schema = load_feature_schema()
        report["feature_schema_version"] = feature_schema["version"]
    except Exception as e:
        report["overall_status"] = "FAIL"
        report["recommended_fix"] = f"Failed to load feature schema: {e}"
        return report
    
    # Use Earth as default test case
    if planet_features is None:
        planet_features = {
            "pl_rade": 1.0,
            "pl_masse": 1.0,
            "pl_orbper": 365.25,
            "pl_orbsmax": 1.0,
            "pl_orbeccen": 0.0167,
            "pl_insol": 1.0,
            "pl_eqt": 255.0,
            "pl_dens": 5.51,
            "st_teff": 5778.0,
            "st_mass": 1.0,
            "st_rad": 1.0,
            "st_lum": 1.0
        }
        test_name = "Earth"
    else:
        test_name = planet_features.get("name", "Test Planet")
    
    # =============================================================================
    # CHECK 1: Feature Building & Schema Compliance
    # =============================================================================
    
    check1 = {
        "name": "Feature Building & Schema Compliance",
        "status": "PASS",
        "details": {},
        "issues": []
    }
    
    try:
        # Build features
        features, meta = build_features_v4(planet_features, return_meta=True)
        
        check1["details"]["feature_vector"] = features.tolist()
        check1["details"]["imputed_fields"] = meta["imputed_fields"]
        check1["details"]["warnings"] = meta.get("warnings", [])
        
        # Check for NaNs
        if np.any(np.isnan(features)):
            check1["status"] = "FAIL"
            nan_indices = np.where(np.isnan(features))[0]
            check1["issues"].append(f"NaN values at indices: {nan_indices.tolist()}")
            report["overall_status"] = "FAIL"
            report["recommended_fix"] = "Feature building produced NaN values"
        
        # Validate against schema
        validation_warnings = validate_features(features, feature_schema)
        if validation_warnings:
            check1["status"] = "WARN"
            check1["issues"].extend(validation_warnings)
            if report["overall_status"] == "PASS":
                report["overall_status"] = "WARN"
                report["recommended_fix"] = "Some features outside expected ranges"
    
    except Exception as e:
        check1["status"] = "FAIL"
        check1["issues"].append(f"Feature building failed: {str(e)}")
        report["overall_status"] = "FAIL"
        report["recommended_fix"] = f"Feature building error: {str(e)}"
    
    report["checks"]["feature_building"] = check1
    
    # =============================================================================
    # CHECK 2: Model Inference
    # =============================================================================
    
    check2 = {
        "name": "Model Inference",
        "status": "PASS",
        "details": {},
        "issues": []
    }
    
    if check1["status"] != "FAIL":
        try:
            # Test prediction
            score = ml_calculator.predict(planet_features, return_raw=False)
            raw_score = ml_calculator.predict(planet_features, return_raw=True)
            
            check2["details"]["earth_normalized_score"] = float(score)
            check2["details"]["raw_score"] = float(raw_score)
            check2["details"]["interpretation"] = "0-100 scale (Earth=100), NOT probability of life"
            
            # Check score range
            if score < 0 or score > 100:
                check2["status"] = "FAIL"
                check2["issues"].append(f"Score {score:.2f}% outside valid range [0, 100]")
                report["overall_status"] = "FAIL"
                report["recommended_fix"] = "Model output normalization issue"
            
            if raw_score < 0 or raw_score > 1:
                check2["status"] = "FAIL"
                check2["issues"].append(f"Raw score {raw_score:.4f} outside valid range [0, 1]")
                report["overall_status"] = "FAIL"
                report["recommended_fix"] = "Model output range issue"
        
        except Exception as e:
            check2["status"] = "FAIL"
            check2["issues"].append(f"Prediction failed: {str(e)}")
            report["overall_status"] = "FAIL"
            report["recommended_fix"] = f"Model inference error: {str(e)}"
    else:
        check2["status"] = "SKIPPED"
        check2["issues"].append("Skipped due to feature building failure")
    
    report["checks"]["model_inference"] = check2
    
    # =============================================================================
    # CHECK 3: Solar System Validation
    # =============================================================================
    
    check3 = {
        "name": "Solar System Ranking Test",
        "status": "PASS",
        "details": {
            "scores": {},
            "ranking": []
        },
        "issues": []
    }
    
    try:
        # Run Solar System validation
        validation_report = validate_solar_system_ranking(
            predict_fn=lambda f: ml_calculator._predict_raw(f),
            feature_builder_fn=build_features_v4,
            export_path=None  # Don't export here
        )
        
        check3["details"]["scores"] = validation_report["scores"]
        check3["details"]["ranking"] = sorted(
            validation_report["scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Check gates
        if not validation_report["all_pass"]:
            check3["status"] = "FAIL"
            for gate_name, gate_data in validation_report["gates"].items():
                if not gate_data["pass"]:
                    check3["issues"].append(f"{gate_name}: {gate_data['description']}")
            report["overall_status"] = "FAIL"
            report["recommended_fix"] = "Solar System ranking incorrect - model calibration issue"
        else:
            check3["details"]["all_gates_passed"] = True
    
    except Exception as e:
        check3["status"] = "FAIL"
        check3["issues"].append(f"Solar System validation failed: {str(e)}")
        if report["overall_status"] == "PASS":
            report["overall_status"] = "WARN"
    
    report["checks"]["solar_system_validation"] = check3
    
    # =============================================================================
    # CHECK 4: Feature Importance & Explainability
    # =============================================================================
    
    check4 = {
        "name": "Feature Importance & Explainability",
        "status": "PASS",
        "details": {},
        "issues": []
    }
    
    if check1["status"] != "FAIL" and check2["status"] != "FAIL":
        try:
            explanation = ml_calculator.explain_prediction(planet_features)
            
            # Get top 5 features
            sorted_importances = sorted(
                explanation["feature_importances"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            check4["details"]["top_features"] = [
                {
                    "name": name,
                    "importance": float(imp),
                    "value": float(explanation["feature_values"][name])
                }
                for name, imp in sorted_importances[:5]
            ]
            
            check4["details"]["imputed_fields"] = explanation["imputed_fields"]
        
        except Exception as e:
            check4["status"] = "WARN"
            check4["issues"].append(f"Failed to get explanation: {str(e)}")
    else:
        check4["status"] = "SKIPPED"
        check4["issues"].append("Skipped due to previous failures")
    
    report["checks"]["explainability"] = check4
    
    # =============================================================================
    # Export Report
    # =============================================================================
    
    try:
        os.makedirs(export_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(export_dir, f"sanity_check_v4_{timestamp}.json")
        
        with open(export_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        report["export_path"] = export_path
        print(f"\n{'='*70}")
        print(f"ML v4 SANITY CHECK COMPLETE")
        print(f"{'='*70}")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Recommended Fix: {report['recommended_fix']}")
        print(f"Report exported to: {export_path}")
        print(f"{'='*70}\n")
    
    except Exception as e:
        report["export_error"] = str(e)
        print(f"Warning: Could not export report: {e}")
    
    return report


if __name__ == "__main__":
    # Test sanity check with mock calculator
    print("Testing ML v4 sanity check...")
    
    try:
        from ml_habitability_v4 import MLHabitabilityCalculatorV4
        
        # Initialize calculator
        calc = MLHabitabilityCalculatorV4()
        
        # Run sanity check
        report = run_ml_sanity_check_v4(
            ml_calculator=calc,
            planet_features=None,  # Use Earth
            export_dir="../exports"
        )
        
        if report["overall_status"] == "PASS":
            print("\n[SUCCESS] All sanity checks passed")
            exit(0)
        else:
            print(f"\n[{report['overall_status']}] Sanity checks completed with issues")
            print(f"Recommended fix: {report['recommended_fix']}")
            exit(1 if report["overall_status"] == "FAIL" else 0)
    
    except Exception as e:
        print(f"\n[ERROR] Sanity check failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
