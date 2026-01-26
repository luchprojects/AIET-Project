import torch
import torch.nn as nn
import joblib
import numpy as np
import os
import sys
import json
from datetime import datetime

class HabitabilityModel(nn.Module):
    def __init__(self, input_size):
        super(HabitabilityModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class MLHabitabilityCalculator:
    def __init__(self, model_path=None, scaler_path=None):
        # Handle PyInstaller bundled paths
        if getattr(sys, 'frozen', False):
            # Running as compiled exe
            base_path = sys._MEIPASS
        else:
            # Running as script
            base_path = os.path.join(os.path.dirname(__file__), '..')
        
        if model_path is None:
            # Try v3 model first (temperature-aware), then v2, then v1
            model_path_v3 = os.path.join(base_path, 'ml_calibration', 'hab_net_v3.pth')
            model_path_v2 = os.path.join(base_path, 'ml_calibration', 'hab_net_v2_fixed.pth')
            model_path_v1 = os.path.join(base_path, 'ml_calibration', 'hab_net_v1.pth')
            if os.path.exists(model_path_v3):
                model_path = model_path_v3
            elif os.path.exists(model_path_v2):
                model_path = model_path_v2
            else:
                model_path = model_path_v1
        if scaler_path is None:
            # Try v3 scaler first, then v2, then v1
            scaler_path_v3 = os.path.join(base_path, 'ml_calibration', 'scaler_v3.joblib')
            scaler_path_v2 = os.path.join(base_path, 'ml_calibration', 'scaler_v2_fixed.joblib')
            scaler_path_v1 = os.path.join(base_path, 'ml_calibration', 'scaler_v1.joblib')
            if os.path.exists(scaler_path_v3):
                scaler_path = scaler_path_v3
            elif os.path.exists(scaler_path_v2):
                scaler_path = scaler_path_v2
            else:
                scaler_path = scaler_path_v1
            
        # v3 feature list with derived physics + TEMPERATURE
        self.feature_cols = [
            "pl_rade", "pl_masse", "pl_orbper", "pl_orbeccen",
            "pl_insol", "pl_dens", "v_esc_kms", "tidal_lock_score",
            "T_surf",  # ADDED: Surface temperature (critical for habitability!)
            "st_teff", "st_mass", "st_rad", "st_lum"
        ]
        
        # CALIBRATION: Compute Earth's raw score to normalize all predictions
        # This ensures Earth always scores exactly 100%
        self.earth_reference_score = None
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            
        self.model = HabitabilityModel(len(self.feature_cols))
        if os.path.exists(model_path):
            # Load the state dict and fix key naming mismatch
            state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
            
            # Check if keys need "model." prefix (trained model was nn.Sequential, not wrapped in class)
            if "0.weight" in state_dict and "model.0.weight" not in state_dict:
                # Add "model." prefix to all keys to match our class structure
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_state_dict[f"model.{key}"] = value
                state_dict = new_state_dict
            
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model.eval()
        
        # Calibrate to Earth reference (ensures Earth = 100%)
        self._calibrate_to_earth()

    def _calibrate_to_earth(self):
        """
        Compute Earth's raw score to use as calibration reference.
        This ensures Earth always scores exactly 100%.
        """
        earth_features = {
            "pl_rade": 1.0,
            "pl_masse": 1.0,
            "pl_orbper": 365.25,
            "pl_orbeccen": 0.0167,
            "pl_insol": 1.0,
            "st_teff": 5778.0,
            "st_mass": 1.0,
            "st_rad": 1.0,
            "st_lum": 1.0
        }
        
        try:
            # Get Earth's raw score (without normalization)
            derived = self.compute_derived_features(earth_features)
            complete_features = {**earth_features, **derived}
            x = np.array([[complete_features[col] for col in self.feature_cols]], dtype=np.float32)
            x_scaled = self.scaler.transform(x)
            
            with torch.no_grad():
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
                self.earth_reference_score = self.model(x_tensor).item()
            
            print(f"[ML Calibration] Earth raw score: {self.earth_reference_score:.2f}%")
            print(f"[ML Calibration] All scores will be normalized so Earth = 100%")
        except Exception as e:
            print(f"Warning: Could not calibrate to Earth: {e}")
            self.earth_reference_score = None

    def compute_derived_features(self, features: dict) -> dict:
        """
        Compute missing features that the teacher formula uses.
        Adds: pl_dens (density), v_esc_kms (escape velocity), tidal_lock_score, T_surf (surface temp)
        """
        # Constants
        G = 6.67430e-11  # m^3 kg^-1 s^-2
        M_earth = 5.972e24  # kg
        R_earth = 6.371e6  # m
        
        # Get base values
        pl_masse = features.get("pl_masse", 1.0)  # Earth masses
        pl_rade = features.get("pl_rade", 1.0)    # Earth radii
        pl_orbper = features.get("pl_orbper", 365.25)  # days
        pl_insol = features.get("pl_insol", 1.0)  # Stellar flux
        
        # Convert to SI
        mass_kg = pl_masse * M_earth
        radius_m = pl_rade * R_earth
        
        # Compute density (g/cm³)
        volume_m3 = (4/3) * np.pi * (radius_m ** 3)
        density_gcm3 = (mass_kg / volume_m3) / 1000.0 if volume_m3 > 0 else 5.51
        
        # Compute escape velocity (km/s)
        v_esc_ms = np.sqrt(2 * G * mass_kg / radius_m) if radius_m > 0 else 0.0
        v_esc_kms = v_esc_ms / 1000.0
        
        # Compute tidal locking score (sigmoid centered at 25 days)
        # FIXED: Shorter periods → more locked (score ~1), longer periods → less locked (score ~0)
        tidal_lock_score = 1.0 / (1.0 + np.exp((pl_orbper - 25.0) / 5.0))
        
        # Surface temperature: Use runtime temperature first, then NASA pl_eqt, then estimate
        # FIXED: Check "temperature" (simulation) first, then "pl_eqt" (NASA), then fallback
        if "temperature" in features:
            T_surf = float(features["temperature"])
        elif "pl_eqt" in features:
            T_surf = float(features["pl_eqt"])
        else:
            # Fallback: compute from equilibrium + greenhouse estimate
            T_eq = 279.0 * (pl_insol ** 0.25)
            greenhouse_offset = 33.0 if pl_rade < 1.6 else 70.0
            T_surf = T_eq + greenhouse_offset
        
        return {
            "pl_dens": float(density_gcm3),
            "v_esc_kms": float(v_esc_kms),
            "tidal_lock_score": float(tidal_lock_score),
            "T_surf": float(T_surf)
        }

    def predict(self, features: dict) -> float:
        """
        Predict habitability percentage based on physical features.
        Expected features dict keys: pl_rade, pl_masse, pl_orbper, pl_orbeccen, pl_insol, st_teff, st_mass, st_rad, st_lum
        v2 automatically computes: pl_dens, v_esc_kms, tidal_lock_score
        
        Scores are normalized so Earth = 100% exactly.
        """
        try:
            # Compute derived features (density, escape velocity, tidal locking)
            derived = self.compute_derived_features(features)
            
            # Merge with input features
            complete_features = {**features, **derived}
            
            # Prepare feature vector in the correct order
            x = np.array([[complete_features[col] for col in self.feature_cols]], dtype=np.float32)
            x_scaled = self.scaler.transform(x)
            
            with torch.no_grad():
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
                raw_prediction = self.model(x_tensor).item()
            
            # Normalize to Earth reference (Earth = 100%)
            if self.earth_reference_score and self.earth_reference_score > 0:
                normalized_prediction = (raw_prediction / self.earth_reference_score) * 100.0
            else:
                normalized_prediction = raw_prediction
                
            # Cap at 100% - Earth is the maximum habitable reference
            return float(np.clip(normalized_prediction, 0.0, 100.0))
        except Exception as e:
            print(f"Error in habitability prediction: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


def run_ml_sanity_check(ml_calculator, planet_features, star_features=None, export_dir="exports"):
    """
    Run comprehensive ML sanity checks to diagnose low Earth scores.
    
    Args:
        ml_calculator: MLHabitabilityCalculator instance
        planet_features: Dict with planet parameters (already combined with star if needed)
        star_features: Optional separate star dict for logging clarity
        export_dir: Directory to export JSON report
    
    Returns:
        dict: Comprehensive diagnostic report with PASS/WARN/FAIL status
    """
    
    # Initialize report
    report = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "overall_status": "PASS",
        "recommended_fix": "No issues detected"
    }
    
    # Expected feature schema from training (hardcoded reference)
    EXPECTED_FEATURES = [
        "pl_rade", "pl_masse", "pl_orbper", "pl_orbeccen",
        "pl_insol", "st_teff", "st_mass", "st_rad", "st_lum"
    ]
    
    EXPECTED_UNITS = {
        "pl_rade": "Earth radii (R⊕)",
        "pl_masse": "Earth masses (M⊕)",
        "pl_orbper": "days",
        "pl_orbeccen": "unitless [0-1]",
        "pl_insol": "Earth flux units",
        "st_teff": "Kelvin (K)",
        "st_mass": "Solar masses (M☉)",
        "st_rad": "Solar radii (R☉)",
        "st_lum": "Solar luminosities (L☉)"
    }
    
    EXPECTED_RANGES = {
        "pl_rade": (0.1, 20.0),
        "pl_masse": (0.001, 500.0),
        "pl_orbper": (0.1, 100000.0),
        "pl_orbeccen": (0.0, 1.0),
        "pl_insol": (0.0001, 100.0),
        "st_teff": (2000.0, 50000.0),
        "st_mass": (0.08, 100.0),
        "st_rad": (0.1, 1000.0),
        "st_lum": (0.0001, 1000000.0)
    }
    
    # ========================================================================
    # CHECK 1: Input Sanity (Earth/Sun reference values)
    # ========================================================================
    check1 = {
        "name": "Input Sanity Check",
        "status": "PASS",
        "details": {},
        "issues": []
    }
    
    # Log all inputs
    check1["details"]["planet_inputs"] = {}
    check1["details"]["star_inputs"] = {}
    
    for key in ["pl_rade", "pl_masse", "pl_orbper", "pl_orbeccen", "pl_insol"]:
        value = planet_features.get(key)
        check1["details"]["planet_inputs"][key] = value
        if value is None or (isinstance(value, float) and np.isnan(value)):
            check1["status"] = "FAIL"
            check1["issues"].append(f"Missing/NaN value for {key}")
    
    for key in ["st_teff", "st_mass", "st_rad", "st_lum"]:
        value = planet_features.get(key)
        check1["details"]["star_inputs"][key] = value
        if value is None or (isinstance(value, float) and np.isnan(value)):
            check1["status"] = "FAIL"
            check1["issues"].append(f"Missing/NaN value for {key}")
    
    if check1["status"] == "FAIL":
        report["overall_status"] = "FAIL"
        report["recommended_fix"] = "Fix missing/NaN input values - check fill policy"
    
    report["checks"]["input_sanity"] = check1
    
    # ========================================================================
    # CHECK 2: Unit/Range Validation
    # ========================================================================
    check2 = {
        "name": "Unit & Range Validation",
        "status": "PASS",
        "details": {},
        "issues": []
    }
    
    for feature in EXPECTED_FEATURES:
        value = planet_features.get(feature)
        expected_unit = EXPECTED_UNITS.get(feature, "unknown")
        expected_range = EXPECTED_RANGES.get(feature, (None, None))
        
        check2["details"][feature] = {
            "value": float(value) if value is not None else None,
            "expected_unit": expected_unit,
            "expected_range": expected_range,
            "in_range": None
        }
        
        if value is not None:
            if expected_range[0] is not None and (value < expected_range[0] or value > expected_range[1]):
                check2["status"] = "WARN"
                check2["issues"].append(
                    f"{feature} = {value:.4f} outside expected range {expected_range} ({expected_unit})"
                )
                check2["details"][feature]["in_range"] = False
                if report["overall_status"] == "PASS":
                    report["overall_status"] = "WARN"
                    report["recommended_fix"] = "Check unit conversions - values outside expected ranges"
            else:
                check2["details"][feature]["in_range"] = True
    
    report["checks"]["unit_range_validation"] = check2
    
    # ========================================================================
    # CHECK 3: Feature Schema Validation
    # ========================================================================
    check3 = {
        "name": "Feature Schema Check",
        "status": "PASS",
        "details": {
            "expected_features": EXPECTED_FEATURES,
            "runtime_features": ml_calculator.feature_cols if ml_calculator else [],
            "match": None
        },
        "issues": []
    }
    
    if ml_calculator:
        if ml_calculator.feature_cols != EXPECTED_FEATURES:
            check3["status"] = "FAIL"
            check3["issues"].append("Feature order/names mismatch between runtime and expected schema")
            check3["details"]["match"] = False
            report["overall_status"] = "FAIL"
            report["recommended_fix"] = "Feature schema mismatch - check feature_cols order"
        else:
            check3["details"]["match"] = True
    else:
        check3["status"] = "FAIL"
        check3["issues"].append("ML calculator not initialized")
        check3["details"]["match"] = False
        report["overall_status"] = "FAIL"
        report["recommended_fix"] = "ML calculator initialization failed"
    
    report["checks"]["feature_schema"] = check3
    
    # ========================================================================
    # CHECK 4: Model Inference & Output Interpretation
    # ========================================================================
    check4 = {
        "name": "Model Output Check",
        "status": "WARN",  # Always warn about interpretation
        "details": {},
        "issues": ["WARN: ML score is uncalibrated - not a true probability of life"]
    }
    
    if ml_calculator and check1["status"] != "FAIL":
        try:
            raw_output = ml_calculator.predict(planet_features)
            check4["details"]["raw_model_output"] = float(raw_output)
            check4["details"]["clipped_output"] = float(np.clip(raw_output, 0.0, 100.0))
            check4["details"]["interpretation"] = "Relative habitability score (0-100), NOT probability"
        except Exception as e:
            check4["status"] = "FAIL"
            check4["issues"].append(f"Prediction failed: {str(e)}")
            report["overall_status"] = "FAIL"
            report["recommended_fix"] = "Model prediction failed - check feature preparation"
    
    report["checks"]["model_output"] = check4
    
    # ========================================================================
    # CHECK 5: Solar System Ranking Test
    # ========================================================================
    check5 = {
        "name": "Solar System Ranking Test",
        "status": "PASS",
        "details": {
            "scores": {},
            "ranking": []
        },
        "issues": []
    }
    
    if ml_calculator and check1["status"] != "FAIL":
        # Solar System planet reference values (in ML feature units)
        solar_system_planets = {
            "Mercury": {
                "pl_rade": 0.383, "pl_masse": 0.055, "pl_orbper": 88.0, "pl_orbeccen": 0.2056,
                "pl_insol": 6.67, "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
            },
            "Venus": {
                "pl_rade": 0.949, "pl_masse": 0.815, "pl_orbper": 225.0, "pl_orbeccen": 0.0068,
                "pl_insol": 1.91, "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
            },
            "Earth": {
                "pl_rade": 1.0, "pl_masse": 1.0, "pl_orbper": 365.25, "pl_orbeccen": 0.0167,
                "pl_insol": 1.0, "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
            },
            "Mars": {
                "pl_rade": 0.532, "pl_masse": 0.107, "pl_orbper": 687.0, "pl_orbeccen": 0.0934,
                "pl_insol": 0.43, "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
            },
            "Jupiter": {
                "pl_rade": 11.2, "pl_masse": 317.8, "pl_orbper": 4333.0, "pl_orbeccen": 0.0484,
                "pl_insol": 0.037, "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
            }
        }
        
        try:
            for planet_name, features in solar_system_planets.items():
                score = ml_calculator.predict(features)
                check5["details"]["scores"][planet_name] = float(score)
            
            # Sort by score
            ranking = sorted(check5["details"]["scores"].items(), key=lambda x: x[1], reverse=True)
            check5["details"]["ranking"] = [{"planet": name, "score": score} for name, score in ranking]
            
            # Check if Earth is in top 2
            earth_rank = next((i for i, (name, _) in enumerate(ranking) if name == "Earth"), None)
            if earth_rank is not None and earth_rank >= 2:
                check5["status"] = "WARN"
                check5["issues"].append(
                    f"Earth ranked #{earth_rank + 1} (expected top 2) - likely unit or label mismatch"
                )
                if report["overall_status"] == "PASS":
                    report["overall_status"] = "WARN"
                    report["recommended_fix"] = "Earth ranking low - check training labels and unit consistency"
        
        except Exception as e:
            check5["status"] = "FAIL"
            check5["issues"].append(f"Ranking test failed: {str(e)}")
    
    report["checks"]["solar_system_ranking"] = check5
    
    # ========================================================================
    # Export to JSON
    # ========================================================================
    try:
        os.makedirs(export_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(export_dir, f"debug_ml_{timestamp}.json")
        
        with open(export_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        report["export_path"] = export_path
        print(f"\n{'='*70}")
        print(f"ML SANITY CHECK COMPLETE")
        print(f"{'='*70}")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Recommended Fix: {report['recommended_fix']}")
        print(f"Report exported to: {export_path}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        report["export_error"] = str(e)
        print(f"Warning: Could not export report: {e}")
    
    return report
