# ==========================================================
# AIET Exoplanet Pipeline v2: FIXED Feature Alignment
# ==========================================================
# Fix: Add density, escape velocity, and tidal locking to ML features

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# CONFIGURATION
# -------------------------
BASE_DIR = r"C:\Users\LuchK\AIET\AIET_ML_inputs"
PLANETS_FILE = os.path.join(BASE_DIR, "exoplanets.csv")
STARS_FILE   = os.path.join(BASE_DIR, "stellar_hosts.csv")

OUTPUT_FILE        = os.path.join(BASE_DIR, "MLP_v2_scored.csv")
MODEL_SAVE_PATH    = os.path.join(BASE_DIR, "hab_net_v2_fixed.pth")
SCALER_SAVE_PATH   = os.path.join(BASE_DIR, "scaler_v2_fixed.joblib")

# -------------------------
# LOAD DATA
# -------------------------
print(">>> Loading Data...")
planets_df = pd.read_csv(PLANETS_FILE, comment="#", low_memory=False)
stars_df   = pd.read_csv(STARS_FILE, comment="#", low_memory=False)

if "default_flag" in planets_df.columns:
    planets_df = planets_df[planets_df["default_flag"] == 1]

df = pd.merge(planets_df, stars_df, on="hostname", how="left")
print(f"Data Loaded: {len(df)} exoplanets.")

# ==========================================================
# PHASE 1 — ASTROPHYSICAL IMPUTATION
# ==========================================================
print(">>> Phase 1: Cleaning Data with Astrophysics...")

bins   = [0, 3700, 5200, 6000, 7500, 50000]
labels = ["M", "K", "G", "F", "Hot"]
df["spectral_bin"] = pd.cut(df["st_teff"], bins=bins, labels=labels)

for col in ["st_mass", "st_rad", "st_lum"]:
    df[col] = df[col].fillna(
        df.groupby("spectral_bin", observed=True)[col].transform("median")
    )
    df[col] = df[col].fillna(df[col].median())

# Stefan–Boltzmann enforcement
df["st_lum"] = df["st_lum"].fillna(
    (df["st_rad"]**2) * (df["st_teff"] / 5778.0)**4
)

# Planet radius and orbital parameters
df["pl_rade"]   = df["pl_rade"].fillna(df["pl_rade"].median())
df["pl_orbper"] = df["pl_orbper"].fillna(df["pl_orbper"].median())
df["pl_orbeccen"] = df["pl_orbeccen"].fillna(0.0)

# Rocky-only mass-radius relation
mask_rocky = (df["pl_rade"] < 1.6) & (df["pl_masse"].isna())
df.loc[mask_rocky, "pl_masse"] = df.loc[mask_rocky, "pl_rade"]**2.06

# ==========================================================
# PHASE 2 — FEATURE ENGINEERING
# ==========================================================
print(">>> Phase 2: Calculating Derived Physics...")

G        = 6.67430e-11
M_earth  = 5.972e24
R_earth  = 6.371e6

df["pl_mass_kg"]  = df["pl_masse"] * M_earth
df["pl_radius_m"] = df["pl_rade"] * R_earth

# Density (g/cm^3)
df["density_calc"] = (
    df["pl_mass_kg"] / ((4/3) * np.pi * df["pl_radius_m"]**3)
) / 1000.0
df["pl_dens"] = df["pl_dens"].fillna(df["density_calc"])

# Escape velocity (km/s) - convert to km/s for better scaling
df["v_esc_ms"] = np.sqrt(2 * G * df["pl_mass_kg"] / df["pl_radius_m"])
df["v_esc_kms"] = df["v_esc_ms"] / 1000.0  # Convert to km/s

# Semi-major axis (AU)
df["P_yr"] = df["pl_orbper"] / 365.25
df["a_calc"] = (df["P_yr"]**2 * df["st_mass"])**(1/3)
df["pl_orbsmax"] = df["pl_orbsmax"].fillna(df["a_calc"])

# Insolation flux
df["flux_calc"] = df["st_lum"] / (df["pl_orbsmax"]**2)
df["pl_insol"] = df["pl_insol"].fillna(df["flux_calc"])

# Tidal locking score (sigmoid centered at 25 days)
df["tidal_lock_score"] = 1 / (1 + np.exp(-(df["pl_orbper"] - 25)/5))

# ==========================================================
# PHASE 3 — PHYSICS TEACHER (SAME AS v1)
# ==========================================================
print(">>> Phase 3: Generating Ground Truth Scores...")

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

df["Habitability_Physics_Percent"] = (
    0.35 * gaussian(df["pl_insol"], 1.0, 0.4) +
    0.25 * gaussian(df["pl_rade"], 1.0, 0.3) +
    0.15 * gaussian(df["pl_dens"], 5.5, 2.0) +
    0.15 * gaussian(df["v_esc_kms"], 11.186, 4.0) +  # Using km/s now
    0.10 * df["tidal_lock_score"]
).clip(0,1) * 100

# ==========================================================
# PHASE 4 — ML STUDENT (FIXED FEATURES)
# ==========================================================
print(">>> Phase 4: Training Neural Network with ALIGNED features...")

# FIX: Include the features the teacher uses!
feature_cols = [
    "pl_rade", "pl_masse", "pl_orbper", "pl_orbeccen",
    "pl_insol", "pl_dens", "v_esc_kms", "tidal_lock_score",  # ADDED!
    "st_teff", "st_mass", "st_rad", "st_lum"
]

mask_finite = np.isfinite(df[feature_cols]).all(axis=1) & \
              np.isfinite(df["Habitability_Physics_Percent"])

df_train = df[mask_finite]

X = df_train[feature_cols].values.astype(np.float32)
y = df_train["Habitability_Physics_Percent"].values.astype(np.float32).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f">>> Scaler saved to: {SCALER_SAVE_PATH}")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(len(feature_cols), 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

print("Training...")
for epoch in range(500):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

model.eval()
with torch.no_grad():
    df.loc[mask_finite, "Habitability_ML_Percent"] = (
        model(
            torch.tensor(
                scaler.transform(df.loc[mask_finite, feature_cols].values),
                dtype=torch.float32
            )
        ).numpy().flatten()
    )

# Clip ML outputs to 0–100%
df["Habitability_ML_Percent"] = df["Habitability_ML_Percent"].clip(0,100)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f">>> Model saved to: {MODEL_SAVE_PATH}")

# ==========================================================
# PHASE 5 — TEST ON EARTH
# ==========================================================
print("\n>>> Testing on Earth...")

earth_features = {
    "pl_rade": 1.0,
    "pl_masse": 1.0,
    "pl_orbper": 365.25,
    "pl_orbeccen": 0.0167,
    "pl_insol": 1.0,
    "pl_dens": 5.51,
    "v_esc_kms": 11.186,
    "tidal_lock_score": 1.0,  # Not tidally locked (sigmoid at 365 days ≈ 1.0)
    "st_teff": 5778.0,
    "st_mass": 1.0,
    "st_rad": 1.0,
    "st_lum": 1.0
}

x_earth = np.array([[earth_features[col] for col in feature_cols]], dtype=np.float32)
x_earth_scaled = scaler.transform(x_earth)

with torch.no_grad():
    earth_score = model(torch.tensor(x_earth_scaled, dtype=torch.float32)).item()

print(f"Earth ML Score: {earth_score:.2f}%")
print(f"Expected (Teacher): ~95-100%")

# ==========================================================
# PHASE 6 — EXPORT
# ==========================================================
print(">>> Saving Results...")
df.to_csv(OUTPUT_FILE, index=False)

print("\n>>> Pipeline Complete.")
print(f"If Earth scores > 80%, the fix worked!")
