"""
AIET ML v4 - Runtime Habitability Calculator
Uses XGBoost model with NASA-locked features
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Tuple

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("Warning: XGBoost not installed. ML v4 will not be available.")

from ml_features_v4 import build_features_v4, get_earth_reference_features, load_feature_schema


class MLHabitabilityCalculatorV4:
    """
    ML v4 Habitability Calculator
    
    Key improvements over v3:
    - Uses NASA-locked features (no synthetic T_surf)
    - XGBoost model (more stable than PyTorch MLP)
    - Consistent feature builder for training/inference
    - Clear separation between raw score and Earth-normalized display
    """
    
    def __init__(self, model_path: str = None, schema_path: str = None):
        """
        Initialize ML v4 calculator.
        
        Args:
            model_path: Path to XGBoost model file (hab_xgb_v4.json)
            schema_path: Path to feature schema JSON (features_v4.json)
        """
        
        if xgb is None:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        # Handle PyInstaller bundled paths
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.join(os.path.dirname(__file__), '..')
        
        # Load feature schema
        if schema_path is None:
            schema_path = os.path.join(base_path, 'ml_calibration', 'features_v4.json')
        
        self.feature_schema = load_feature_schema(schema_path)
        self.feature_names = [f["name"] for f in self.feature_schema["features"]]
        
        print(f"[ML v4] Loaded feature schema: {len(self.feature_names)} features")
        
        # Load XGBoost model
        if model_path is None:
            model_path = os.path.join(base_path, 'ml_calibration', 'hab_xgb_v4.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        
        # Load XGBoost model (compatible with XGBoost 3.x)
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        print(f"[ML v4] Loaded XGBoost model from: {model_path}")
        
        # Compute Earth reference for normalization
        self.earth_features, earth_meta = get_earth_reference_features()
        self.earth_raw_score = self._predict_raw(self.earth_features)
        
        print(f"[ML v4] Earth raw score: {self.earth_raw_score:.4f}")
        print(f"[ML v4] Initialization complete")
    
    def _predict_raw(self, features: np.ndarray) -> float:
        """
        Get raw model prediction (0-1 scale).
        
        Args:
            features: Feature vector of shape (12,)
        
        Returns:
            Raw score in [0, 1]
        """
        # Convert to DMatrix for XGBoost Booster
        import xgboost as xgb
        features_2d = features.reshape(1, -1)
        dmatrix = xgb.DMatrix(features_2d)
        raw_score = self.model.predict(dmatrix)[0]
        return float(np.clip(raw_score, 0.0, 1.0))
    
    def predict(
        self,
        features: dict,
        return_raw: bool = False,
        return_meta: bool = False
    ) -> float | Tuple[float, Dict]:
        """
        Predict habitability score from planet/star features.
        
        Args:
            features: Dict with NASA column names (pl_rade, st_teff, etc.)
            return_raw: If True, return raw 0-1 score; if False, return Earth-normalized 0-100
            return_meta: If True, return (score, metadata dict) tuple
        
        Returns:
            Habitability score (0-100 by default, 0-1 if return_raw=True)
            If return_meta=True, returns (score, meta) tuple
        """
        
        try:
            # Build features using canonical builder
            feature_vector, meta = build_features_v4(features, return_meta=True)
            
            # Get raw prediction
            raw_score = self._predict_raw(feature_vector)
            
            # Normalize to Earth if requested
            if return_raw:
                final_score = raw_score
            else:
                # Earth = 100% display mode
                if self.earth_raw_score > 0:
                    normalized_score = (raw_score / self.earth_raw_score) * 100.0
                    final_score = float(np.clip(normalized_score, 0.0, 100.0))
                else:
                    final_score = raw_score * 100.0
            
            # Add prediction info to meta
            meta["raw_score"] = raw_score
            meta["earth_normalized_score"] = final_score if not return_raw else final_score * 100.0
            meta["normalization_mode"] = "raw" if return_raw else "earth_normalized"
            
            if return_meta:
                return final_score, meta
            else:
                return final_score
        
        except Exception as e:
            print(f"[ML v4] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0 if not return_meta else (0.0, {"error": str(e)})
    
    def predict_batch(
        self,
        planet_rows: list,
        return_raw: bool = False
    ) -> np.ndarray:
        """
        Predict scores for multiple planets efficiently.
        
        Args:
            planet_rows: List of dicts with NASA column names
            return_raw: If True, return raw 0-1 scores; if False, Earth-normalized 0-100
        
        Returns:
            Array of scores
        """
        
        # Build feature matrix
        features_list = []
        for row in planet_rows:
            try:
                features, _ = build_features_v4(row, return_meta=False)
                features_list.append(features)
            except Exception as e:
                print(f"[ML v4] Failed to build features for planet: {e}")
                features_list.append(np.zeros(12, dtype=np.float32))
        
        X = np.array(features_list, dtype=np.float32)
        
        # Predict using DMatrix
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        raw_scores = self.model.predict(dmatrix)
        raw_scores = np.clip(raw_scores, 0.0, 1.0)
        
        # Normalize if requested
        if return_raw:
            return raw_scores
        else:
            if self.earth_raw_score > 0:
                normalized_scores = (raw_scores / self.earth_raw_score) * 100.0
                return np.clip(normalized_scores, 0.0, 100.0)
            else:
                return raw_scores * 100.0
    
    def get_earth_score(self, raw: bool = False) -> float:
        """
        Get Earth's reference score.
        
        Args:
            raw: If True, return raw 0-1 score; if False, return 100.0
        
        Returns:
            Earth score
        """
        return self.earth_raw_score if raw else 100.0
    
    def explain_prediction(self, features: dict) -> Dict:
        """
        Get detailed explanation of prediction.
        
        Args:
            features: Dict with NASA column names
        
        Returns:
            Dict with feature values, score, and metadata
        """
        
        # Build features
        feature_vector, meta = build_features_v4(features, return_meta=True)
        
        # Predict
        raw_score = self._predict_raw(feature_vector)
        normalized_score = (raw_score / self.earth_raw_score) * 100.0 if self.earth_raw_score > 0 else raw_score * 100.0
        normalized_score = float(np.clip(normalized_score, 0.0, 100.0))
        
        # Get feature importance from booster
        # For XGBoost Booster, use get_score()
        importance_dict = self.model.get_score(importance_type='weight')
        # Map feature indices (f0, f1, ...) to names
        importances = np.zeros(len(self.feature_names))
        for feat_idx, score in importance_dict.items():
            idx = int(feat_idx.replace('f', ''))
            if idx < len(importances):
                importances[idx] = score
        
        # Normalize importances to sum to 1
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        explanation = {
            "raw_score": float(raw_score),
            "earth_normalized_score": float(normalized_score),
            "feature_values": {
                name: float(feature_vector[i])
                for i, name in enumerate(self.feature_names)
            },
            "feature_importances": {
                name: float(importances[i])
                for i, name in enumerate(self.feature_names)
            },
            "imputed_fields": meta["imputed_fields"],
            "warnings": meta.get("warnings", [])
        }
        
        return explanation


# =============================================================================
# BACKWARDS COMPATIBILITY WRAPPER
# =============================================================================

def get_ml_calculator(version: str = "v4") -> 'MLHabitabilityCalculatorV4':
    """
    Factory function to get ML calculator by version.
    
    Args:
        version: "v4" (default) or "v3" (legacy)
    
    Returns:
        ML calculator instance
    """
    if version == "v4":
        return MLHabitabilityCalculatorV4()
    elif version == "v3":
        from ml_habitability import MLHabitabilityCalculator
        return MLHabitabilityCalculator()
    else:
        raise ValueError(f"Unknown ML version: {version}")


if __name__ == "__main__":
    # Test with Earth
    print("\n" + "="*70)
    print("Testing ML v4 Calculator with Earth")
    print("="*70)
    
    try:
        calc = MLHabitabilityCalculatorV4()
        
        earth_features = {
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
        
        # Test prediction
        score, meta = calc.predict(earth_features, return_raw=False, return_meta=True)
        
        print(f"\nEarth Prediction:")
        print(f"  Earth-normalized score: {score:.2f}%")
        print(f"  Raw score: {meta['raw_score']:.4f}")
        print(f"  Imputed fields: {meta['imputed_fields']}")
        
        # Test explanation
        explanation = calc.explain_prediction(earth_features)
        
        print(f"\nTop 5 Feature Importances:")
        sorted_importances = sorted(
            explanation["feature_importances"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for name, importance in sorted_importances[:5]:
            value = explanation["feature_values"][name]
            print(f"  {name:15s}: {importance:.4f} (value: {value:.4f})")
        
        print("\n[OK] ML v4 calculator test passed")
    
    except Exception as e:
        print(f"\n[ERROR] ML v4 calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
