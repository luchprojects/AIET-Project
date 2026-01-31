# ==========================================================
# AIET Exoplanet Pipeline v3: TEMPERATURE-AWARE Training
# ==========================================================
# Fix: Add surface temperature penalty to teacher formula
# Venus (737K) should score LOW, Earth (288K) should score HIGH

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# -------------------------
# CONFIGURATION
# -------------------------
BASE_DIR = r"C:\Users\LuchK\AIET\AIET_ML_inputs"
PLANETS_FILE = os.path.join(BASE_DIR, "exoplanets.csv")
STARS_FILE   = os.path.join(BASE_DIR, "stellar_hosts.csv")

OUTPUT_FILE        = os.path.join(BASE_DIR, "MLP_v3_scored.csv")
MODEL_SAVE_PATH    = os.path.join(BASE_DIR, "hab_net_v3.pth")
SCALER_SAVE_PATH   = os.path.join(BASE_DIR, "scaler_v3.joblib")

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

# Escape velocity (km/s)
df["v_esc_ms"] = np.sqrt(2 * G * df["pl_mass_kg"] / df["pl_radius_m"])
df["v_esc_kms"] = df["v_esc_ms"] / 1000.0

# Semi-major axis (AU)
df["P_yr"] = df["pl_orbper"] / 365.25
df["a_calc"] = (df["P_yr"]**2 * df["st_mass"])**(1/3)
df["pl_orbsmax"] = df["pl_orbsmax"].fillna(df["a_calc"])

# Insolation flux
df["flux_calc"] = df["st_lum"] / (df["pl_orbsmax"]**2)
df["pl_insol"] = df["pl_insol"].fillna(df["flux_calc"])

# Equilibrium temperature (no greenhouse)
df["T_eq"] = 279.0 * (df["pl_insol"] ** 0.25)  # Earth T_eq = 255K at 1.0 flux

# Estimate surface temperature if not available
# Use equilibrium + typical greenhouse offset based on planet type
df["T_surf_est"] = df["T_eq"].copy()
# Rocky planets (< 1.6 R_earth): add small greenhouse
mask_rocky = df["pl_rade"] < 1.6
df.loc[mask_rocky, "T_surf_est"] = df.loc[mask_rocky, "T_eq"] + 33.0  # Earth-like greenhouse

# Use pl_eqt if available, otherwise use our estimate
if "pl_eqt" in df.columns:
    df["T_surf"] = df["pl_eqt"].fillna(df["T_surf_est"])
else:
    df["T_surf"] = df["T_surf_est"]

# Tidal locking score
df["tidal_lock_score"] = 1 / (1 + np.exp(-(df["pl_orbper"] - 25)/5))

# ==========================================================
# PHASE 3 — PHYSICS TEACHER (WITH TEMPERATURE!)
# ==========================================================
print(">>> Phase 3: Generating Ground Truth Scores with TEMPERATURE penalty...")

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

# NEW: Temperature penalty (centered at 288K, tight tolerance)
df["temp_score"] = gaussian(df["T_surf"], 288.0, 50.0)

# Updated teacher formula with temperature
df["Habitability_Physics_Percent"] = (
    0.30 * gaussian(df["pl_insol"], 1.0, 0.4) +      # Flux (reduced weight)
    0.20 * gaussian(df["pl_rade"], 1.0, 0.3) +       # Size (reduced)
    0.15 * gaussian(df["pl_dens"], 5.5, 2.0) +       # Density
    0.10 * gaussian(df["v_esc_kms"], 11.186, 4.0) +  # Escape velocity
    0.05 * df["tidal_lock_score"] +                  # Tidal locking (reduced)
    0.20 * df["temp_score"]                          # TEMPERATURE (NEW!)
).clip(0,1) * 100

print("Teacher formula updated with temperature penalty:")
print("  30% - Stellar flux")
print("  20% - Planet radius")
print("  15% - Density")
print("  10% - Escape velocity")
print("   5% - Tidal locking")
print("  20% - SURFACE TEMPERATURE (288K optimal)")
print()

# ==========================================================
# PHASE 4 — ML STUDENT
# ==========================================================
print(">>> Phase 4: Training Neural Network...")

feature_cols = [
    "pl_rade", "pl_masse", "pl_orbper", "pl_orbeccen",
    "pl_insol", "pl_dens", "v_esc_kms", "tidal_lock_score",
    "T_surf",  # ADDED: Surface temperature
    "st_teff", "st_mass", "st_rad", "st_lum"
]

mask_finite = np.isfinite(df[feature_cols]).all(axis=1) & \
              np.isfinite(df["Habitability_Physics_Percent"])

df_train = df[mask_finite]
print(f"Training on {len(df_train)} planets with complete data.")

X = df_train[feature_cols].values.astype(np.float32)
y = df_train["Habitability_Physics_Percent"].values.astype(np.float32).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"Scaler saved to: {SCALER_SAVE_PATH}")

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

print("Training for 500 epochs...")
for epoch in range(500):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

model.eval()
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")

# ==========================================================
# PHASE 5 — TEST ON SOLAR SYSTEM
# ==========================================================
print("\n" + "="*70)
print("TESTING ON SOLAR SYSTEM PLANETS")
print("="*70)

solar_system_test = {
    "Mercury": {
        "pl_rade": 0.383, "pl_masse": 0.055, "pl_orbper": 88.0, "pl_orbeccen": 0.2056,
        "pl_insol": 6.67, "pl_dens": 5.43, "v_esc_kms": 4.25, "tidal_lock_score": 0.0,
        "T_surf": 440.0,  # HOT - no atmosphere
        "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
    },
    "Venus": {
        "pl_rade": 0.949, "pl_masse": 0.815, "pl_orbper": 225.0, "pl_orbeccen": 0.0068,
        "pl_insol": 1.91, "pl_dens": 5.24, "v_esc_kms": 10.36, "tidal_lock_score": 0.0,
        "T_surf": 737.0,  # EXTREMELY HOT - runaway greenhouse
        "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
    },
    "Earth": {
        "pl_rade": 1.0, "pl_masse": 1.0, "pl_orbper": 365.25, "pl_orbeccen": 0.0167,
        "pl_insol": 1.0, "pl_dens": 5.51, "v_esc_kms": 11.186, "tidal_lock_score": 1.0,
        "T_surf": 288.0,  # PERFECT - liquid water
        "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
    },
    "Mars": {
        "pl_rade": 0.532, "pl_masse": 0.107, "pl_orbper": 687.0, "pl_orbeccen": 0.0934,
        "pl_insol": 0.43, "pl_dens": 3.93, "v_esc_kms": 5.03, "tidal_lock_score": 1.0,
        "T_surf": 210.0,  # COLD - thin atmosphere
        "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
    },
    "Jupiter": {
        "pl_rade": 11.2, "pl_masse": 317.8, "pl_orbper": 4333.0, "pl_orbeccen": 0.0484,
        "pl_insol": 0.037, "pl_dens": 1.33, "v_esc_kms": 59.5, "tidal_lock_score": 1.0,
        "T_surf": 165.0,  # Gas giant - no solid surface
        "st_teff": 5778.0, "st_mass": 1.0, "st_rad": 1.0, "st_lum": 1.0
    }
}

print("\nSolar System Predictions:")
print("-" * 70)
for planet_name, features in solar_system_test.items():
    x_test = np.array([[features[col] for col in feature_cols]], dtype=np.float32)
    x_test_scaled = scaler.transform(x_test)
    
    with torch.no_grad():
        score = model(torch.tensor(x_test_scaled, dtype=torch.float32)).item()
    
    temp_k = features["T_surf"]
    temp_c = temp_k - 273.15
    print(f"{planet_name:10s}: {score:6.2f}%  (Temp: {temp_k:.0f}K / {temp_c:.0f}°C)")

print("-" * 70)

# Check Earth score
earth_x = np.array([[solar_system_test["Earth"][col] for col in feature_cols]], dtype=np.float32)
earth_x_scaled = scaler.transform(earth_x)
with torch.no_grad():
    earth_score = model(torch.tensor(earth_x_scaled, dtype=torch.float32)).item()

print(f"\nEarth Score: {earth_score:.2f}%")
if earth_score > 85.0:
    print("SUCCESS: Earth scores > 85% with temperature penalty!")
elif earth_score > 70.0:
    print("GOOD: Earth scores > 70% (acceptable)")
else:
    print("WARNING: Earth score still low - may need weight tuning")

# Check Venus score (should be LOW due to extreme temperature)
venus_x = np.array([[solar_system_test["Venus"][col] for col in feature_cols]], dtype=np.float32)
venus_x_scaled = scaler.transform(venus_x)
with torch.no_grad():
    venus_score = model(torch.tensor(venus_x_scaled, dtype=torch.float32)).item()

print(f"Venus Score: {venus_score:.2f}%")
if venus_score < 40.0:
    print("SUCCESS: Venus properly penalized for extreme temperature!")
elif venus_score < earth_score:
    print("GOOD: Venus scores lower than Earth")
else:
    print("WARNING: Venus still scoring too high")

# ==========================================================
# PHASE 6 — EXPORT
# ==========================================================
print("\n>>> Saving Results...")

# Predict on all planets
with torch.no_grad():
    df.loc[mask_finite, "Habitability_ML_Percent"] = (
        model(
            torch.tensor(
                scaler.transform(df.loc[mask_finite, feature_cols].values),
                dtype=torch.float32
            )
        ).numpy().flatten()
    )

df["Habitability_ML_Percent"] = df["Habitability_ML_Percent"].clip(0, 100)
df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "="*70)
print("TRAINING COMPLETE - v3 Model with Temperature Awareness")
print("="*70)
print(f"Model: {MODEL_SAVE_PATH}")
print(f"Scaler: {SCALER_SAVE_PATH}")
print("="*70 + "\n")
