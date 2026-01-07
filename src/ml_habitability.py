import torch
import torch.nn as nn
import joblib
import numpy as np
import os
import sys

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
            # Using weights_only=True for security, but we need to ensure the architecture matches
            self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model.eval()

    def predict(self, features: dict) -> float:
        """
        Predict habitability percentage based on physical features.
        Expected features dict keys: pl_rade, pl_masse, pl_orbper, pl_orbeccen, pl_insol, st_teff, st_mass, st_rad, st_lum
        """
        try:
            # Prepare feature vector in the correct order
            x = np.array([[features[col] for col in self.feature_cols]], dtype=np.float32)
            x_scaled = self.scaler.transform(x)
            
            with torch.no_grad():
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
                prediction = self.model(x_tensor).item()
                
            return float(np.clip(prediction, 0.0, 100.0))
        except Exception as e:
            print(f"Error in habitability prediction: {e}")
            return 0.0
