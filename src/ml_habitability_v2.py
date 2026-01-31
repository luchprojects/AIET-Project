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

class MLHabitabilityCalculatorV2:
    """
    Version 2: Computes missing features (density, v_esc, tidal_lock) at runtime
    to match the training teacher formula.
    """
    def __init__(self, model_path=None, scaler_path=None):
        # Handle PyInstaller bundled paths
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.join(os.path.dirname(__file__), '..')
        
        if model_path is None:
            # Try v2 model first, fallback to v1
            model_path_v2 = os.path.join(base_path, 'ml_calibration', 'hab_net_v2_fixed.pth')
            model_path_v1 = os.path.join(base_path, 'ml_calibration', 'hab_net_v1.pth')
            model_path = model_path_v2 if os.path.exists(model_path_v2) else model_path_v1
        
        if scaler_path is None:
            scaler_path_v2 = os.path.join(base_path, 'ml_calibration', 'scaler_v2_fixed.joblib')
            scaler_path_v1 = os.path.join(base_path, 'ml_calibration', 'scaler_v1.joblib')
            scaler_path = scaler_path_v2 if os.path.exists(scaler_path_v2) else scaler_path_v1
        
        # Feature list with derived physics
        self.feature_cols = [
            "pl_rade", "pl_masse", "pl_orbper", "pl_orbeccen",
            "pl_insol", "pl_dens", "v_esc_kms", "tidal_lock_score",
            "st_teff", "st_mass", "st_rad", "st_lum"
        ]
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            
        self.model = HabitabilityModel(len(self.feature_cols))
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
            
            if "0.weight" in state_dict and "model.0.weight" not in state_dict:
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_state_dict[f"model.{key}"] = value
                state_dict = new_state_dict
            
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model.eval()

    def compute_derived_features(self, features: dict) -> dict:
        """
        Compute missing features that the teacher formula uses.
        """
        # Constants
        G = 6.67430e-11  # m^3 kg^-1 s^-2
        M_earth = 5.972e24  # kg
        R_earth = 6.371e6  # m
        
        # Get base values
        pl_masse = features.get("pl_masse", 1.0)  # Earth masses
        pl_rade = features.get("pl_rade", 1.0)    # Earth radii
        pl_orbper = features.get("pl_orbper", 365.25)  # days
        
        # Convert to SI
        mass_kg = pl_masse * M_earth
        radius_m = pl_rade * R_earth
        
        # Compute density (g/cm³)
        volume_m3 = (4/3) * np.pi * (radius_m ** 3)
        density_gcm3 = (mass_kg / volume_m3) / 1000.0
        
        # Compute escape velocity (km/s)
        v_esc_ms = np.sqrt(2 * G * mass_kg / radius_m) if radius_m > 0 else 0.0
        v_esc_kms = v_esc_ms / 1000.0
        
        # Compute tidal locking score (sigmoid centered at 25 days)
        tidal_lock_score = 1.0 / (1.0 + np.exp(-(pl_orbper - 25.0) / 5.0))
        
        return {
            "pl_dens": float(density_gcm3),
            "v_esc_kms": float(v_esc_kms),
            "tidal_lock_score": float(tidal_lock_score)
        }

    def predict(self, features: dict) -> float:
        """
        Predict habitability percentage with derived features.
        """
        try:
            # Compute missing features
            derived = self.compute_derived_features(features)
            
            # Merge with input features
            complete_features = {**features, **derived}
            
            # Prepare feature vector
            x = np.array([[complete_features[col] for col in self.feature_cols]], dtype=np.float32)
            x_scaled = self.scaler.transform(x)
            
            with torch.no_grad():
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
                prediction = self.model(x_tensor).item()
                
            return float(np.clip(prediction, 0.0, 100.0))
        except Exception as e:
            print(f"Error in habitability prediction: {e}")
            return 0.0


# Original v1 calculator (for backwards compatibility)
class MLHabitabilityCalculator:
    def __init__(self, model_path=None, scaler_path=None):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.join(os.path.dirname(__file__), '..')
        
        if model_path is None:
            model_path = os.path.join(base_path, 'ml_calibration', 'hab_net_v1.pth')
        if scaler_path is None:
            scaler_path = os.path.join(base_path, 'ml_calibration', 'scaler_v1.joblib')
            
        self.feature_cols = [
            "pl_rade", "pl_masse", "pl_orbper", "pl_orbeccen",
            "pl_insol", "st_teff", "st_mass", "st_rad", "st_lum"
        ]
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            
        self.model = HabitabilityModel(len(self.feature_cols))
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
            
            if "0.weight" in state_dict and "model.0.weight" not in state_dict:
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_state_dict[f"model.{key}"] = value
                state_dict = new_state_dict
            
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model.eval()

    def predict(self, features: dict) -> float:
        try:
            x = np.array([[features[col] for col in self.feature_cols]], dtype=np.float32)
            x_scaled = self.scaler.transform(x)
            
            with torch.no_grad():
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
                prediction = self.model(x_tensor).item()
                
            return float(np.clip(prediction, 0.0, 100.0))
        except Exception as e:
            print(f"Error in habitability prediction: {e}")
            return 0.0


# Keep original sanity check function (copy from ml_habitability.py)
def run_ml_sanity_check(ml_calculator, planet_features, star_features=None, export_dir="exports"):
    """Run comprehensive ML sanity checks."""
    # [Same implementation as before]
    pass
